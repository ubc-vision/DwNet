import torch
import torch.nn as nn
import functools
from torch.nn.functional import grid_sample, interpolate
from torch.autograd import Variable
import numpy as np
import util.util as util

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, loadSize, batchSize, ngf, n_downsample_global=3, n_blocks_global=9,
             n_blocks_local=3, norm='instance', gpu_ids=[], grid_padding = 'reflection', no_coarse_warp= False, no_refining_warp = False):
    norm_layer = get_norm_layer(norm_type=norm)
    netG = GlobalGenerator(input_nc, output_nc, loadSize, batchSize, ngf, n_downsample_global, n_blocks_global,
                                norm_layer, grid_padding = grid_padding, no_coarse_warp =  no_coarse_warp, no_refining_warp = no_refining_warp)
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    netG.model_downsample_grid_set[12].weight.data.fill_(0.0)
    netG.model_downsample_grid_set[12].bias.data.fill_(0.0)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class WarpLoss(nn.Module):
    def __init__(self, gpu_ids, padding_mode):
        super(WarpLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.padding_mode = padding_mode

    def forward(self, grid_source, grid_prev, source, previous, real):
        size = list(grid_source.shape)[1:3]
        source = interpolate(source, size, mode='bilinear')
        previous = interpolate(previous,  size,  mode='bilinear')
        real = interpolate(real, size, mode='bilinear')
        warped_source = grid_sample(source, grid_source, padding_mode=self.padding_mode)
        warped_previous = grid_sample(previous, grid_prev, padding_mode=self.padding_mode)
        x1_vgg, x2_vgg, y_vgg = self.vgg(warped_source), self.vgg(warped_previous), self.vgg(real)
        loss = 0
        for i in range(len(x1_vgg)):
            loss += self.weights[i] * self.criterion(x1_vgg[i], y_vgg[i])
            loss += self.weights[i] * self.criterion(x2_vgg[i], y_vgg[i].detach())
        return loss


##############################################################################
# Generator
##############################################################################
class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, loadSize, batchSize, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', grid_padding = 'reflection', no_coarse_warp = False, no_refining_warp = False):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)
        self.loadSize = loadSize
        self.batchSize = batchSize
        self.grid_padding = grid_padding
        self.no_coarse_warp = no_coarse_warp
        self.no_refining_warp = no_refining_warp
        print "INPUT NUMBER OF CHANNELS", input_nc

        model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model_downsample += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
        self.model_downsample = nn.Sequential(*model_downsample)


        model_downsample_previous = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample_previous
        for i in range(n_downsampling):
            mult = 2**i
            model_downsample_previous += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
        self.model_downsample_previous = nn.Sequential(*model_downsample_previous)


        ### downsample_grid_set
        model_downsample_grid_set = [nn.ReflectionPad2d(3), nn.Conv2d(8, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            model_downsample_grid_set += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                                 norm_layer(ngf * mult * 2), activation]
        mult = 2 ** n_downsampling
        ## adding 2 resnet blocks
        for i in range(1, 3):
            model_downsample_grid_set += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        model_downsample_grid_set += [nn.Conv2d(ngf*mult, 2, kernel_size=1)]
        self.model_downsample_grid_set = nn.Sequential(*model_downsample_grid_set)

        ## convolution layer to decrease number of channels after concatenation
        self.down_pair = nn.Conv2d(3*ngf*mult,ngf*mult, kernel_size=1)

        ### resnet blocks
        model = [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        for i in range(1, n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def warp_module(self, frame, grid, dp_target):
        ### grid in the relative coordinates'
        grid_output = interpolate(grid, size= (64, 64), mode='bilinear')
        if not self.no_refining_warp:
            rel_coord_grid = grid - util.make_coordinate_grid((self.loadSize, self.loadSize), type(grid), self.batchSize).cuda()
            warped_source = grid_sample(frame, grid.permute(0, 2, 3, 1), padding_mode = self.grid_padding )
            grid_set = torch.cat([dp_target, warped_source, rel_coord_grid], dim = 1)
            grid_set_output = self.model_downsample_grid_set(grid_set)
            grid_output_global = (grid_set_output + grid_output).permute(0, 2, 3, 1)
        else:
            grid_output_global = grid_output.permute(0, 2, 3, 1)
        return grid_output_global


    def forward(self, dp_target, source_frame, prev_frame, grid_source, grid_prev):
        model_downsample_output = self.model_downsample(dp_target)

        ## previous frame preprocess
        prev_frame_output = self.model_downsample_previous(prev_frame)
        if not self.no_coarse_warp:
            grid_for_prev = self.warp_module(prev_frame, grid_prev, dp_target)
            warped_prev = grid_sample(prev_frame_output, grid_for_prev, padding_mode= self.grid_padding )
        else:
            warped_prev = prev_frame_output

        ## source frame warping
        source_frame_output = self.model_downsample_previous(source_frame)
        if not self.no_coarse_warp:
            grid_for_source = self.warp_module(source_frame, grid_source, dp_target)
            warped_source = grid_sample(source_frame_output, grid_for_source, padding_mode= self.grid_padding )
        else:
            warped_source = source_frame_output

        output = torch.cat((model_downsample_output, warped_source, warped_prev), 1)
        output = self.down_pair(output)
        return self.model(output), grid_for_source if not self.no_coarse_warp else None, grid_for_prev if not self.no_coarse_warp else None

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
