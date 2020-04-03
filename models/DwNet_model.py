import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class DwNetModel(BaseModel):
    def name(self):
        return 'DwNetModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_warp_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, use_warp_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, g_warp, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg, g_warp, d_real,d_fake),flags) if f]
        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.input_nc

        ##### define networks
        # Generator network
        netG_input_nc = input_nc

        self.netG = networks.define_G(netG_input_nc, opt.output_nc,opt.loadSize, opt.batchSize, opt.ngf,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_blocks_local,
                                      opt.norm, gpu_ids=self.gpu_ids, grid_padding = opt.grid_padding,
                                      no_coarse_warp = opt.no_coarse_warp, no_refining_warp = opt.no_refining_warp)

        # Discriminator network
        if self.isTrain:
            netD_input_nc = input_nc + opt.output_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not (opt.no_warp_loss or opt.no_coarse_warp or opt.no_refining_warp))

            self.criterionGAN = networks.GANLoss(tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionWarp = networks.WarpLoss(self.gpu_ids, opt.grid_padding)

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG', 'G_Warp', 'D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, dp_target, source_frame, prev_frame, grid_source, grid_prev, inst_map=None, real_image=None, feat_map=None, infer=False):

        input_label = Variable(dp_target.data.cuda())
        source_frame =  Variable(source_frame.data.cuda())
        prev_frame = Variable(prev_frame.data.cuda())
        grid_source = Variable(grid_source.data.cuda())
        grid_prev = Variable(grid_prev.data.cuda())

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        return input_label, source_frame, prev_frame, grid_source, grid_prev, real_image

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, dp_target, source_frame, prev_frame, grid_source, grid_prev, image, inst=None,  feat=None, infer=False):
        # Encode Inputs
        input_label, source_frame, prev_frame, grid_source, grid_prev, real_image = self.encode_input(
                                                                            dp_target,
                                                                            source_frame, prev_frame,
                                                                            grid_source, grid_prev,
                                                                            real_image = image)

        fake_image, grid_for_source, grid_for_prev = self.netG.forward(input_label,
                                                                  source_frame,
                                                                  prev_frame,
                                                                  grid_source,
                                                                  grid_prev)
        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat

        if self.opt.no_coarse_warp or self.opt.no_refining_warp:
            loss_G_Warp = None
        else:
            loss_G_Warp = self.criterionWarp( grid_for_source, grid_for_prev, source_frame, prev_frame, real_image)

        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_Warp, loss_D_real, loss_D_fake ), fake_image, grid_for_source, grid_for_prev ]

    def inference(self, dp_target, source_frame, prev_frame, grid_source, grid_prev):
        # Encode Inputs
        input_label, source_frame, prev_frame, grid_source, grid_prev, real_image = self.encode_input(
                                                                            dp_target,
                                                                            source_frame, prev_frame,
                                                                            grid_source, grid_prev, infer=True)

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image, grid_result, _ = self.netG.forward(input_label,
                                                                          source_frame,
                                                                          prev_frame,
                                                                          grid_source,
                                                                          grid_prev)
        else:
            fake_image, grid_result, _ = self.netG.forward(input_label,
                                                                      source_frame,
                                                                      prev_frame,
                                                                      grid_source,
                                                                      grid_prev)
        return fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(DwNetModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)
