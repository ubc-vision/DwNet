import time
from collections import OrderedDict
import torch.nn.functional as functional
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.nn.functional import grid_sample
from torch.autograd import Variable

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

lr_descrease_freq = (opt.niter * dataset_size)  //  opt.niter_decay  + 1

print("Frequency of the learning rate decay = %d iterations" % lr_descrease_freq)

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
lr_decrease_delta = total_steps % lr_descrease_freq

for epoch in range(start_epoch, opt.niter + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(dataset, start=epoch_iter):
        data["input"] = data["input"].permute(1, 0, 2, 3, 4)
        data["target"] = data["target"].permute(1, 0, 2, 3, 4)
        data["source_frame"] = data["source_frame"].permute(1, 0, 2, 3, 4)
        data["grid"] = data["grid"].permute(1, 0, 2, 3, 4)
        data["grid_source"] = data["grid_source"].permute(1, 0, 2, 3, 4)

        iter_start_time = time.time()
        epoch_iter += opt.batchSize
        total_steps += opt.batchSize
        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta
        lr_decay = total_steps %  lr_descrease_freq == lr_decrease_delta


        ############## Forward Pass ######################
        losses, generated, grid_for_source, grid_for_prev = model(data['input'][0],
                                        data['source_frame'][0], data['source_frame'][0],
                                        data['grid_source'][0], data['grid_source'][0],
                                        image = data['target'][0], infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) + loss_dict.get('G_Warp',0)

        # update generator weights
        model.module.optimizer_G.zero_grad()
        loss_G.backward()
        model.module.optimizer_G.step()

        # update discriminator weights
        model.module.optimizer_D.zero_grad()
        loss_D.backward()
        model.module.optimizer_D.step()


        ### display output images
        if save_fake:
            img = (data['source_frame'][0]).cuda()
            warped_source = grid_sample(data['source_frame'][0], data['grid_source'][0].permute(0, 2, 3, 1), padding_mode=opt.grid_padding)
            if not (opt.no_coarse_warp or opt.no_refining_warp):
                grid = grid_for_source.permute(0, 3, 1, 2).detach()
                grid = functional.interpolate(grid, (256,256), mode = 'bilinear' )
                warp = functional.grid_sample(img, grid.permute(0, 2, 3, 1), padding_mode=opt.grid_padding)
            else:
                warp = warped_source

            visuals = OrderedDict([('synthesized_video', util.tensor2im(generated.data[0])),
                                    ('source_video', util.tensor2im(data['source_frame'][0][0])),
                                    ('real_video', util.tensor2im(data['target'][0][0])),
                                    ('warped_source', util.tensor2im(warped_source.data[0])),
                                    ('warped_with_learned_grid', util.tensor2im(warp[0]))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        for f in range(1, opt.prev_frame_num):
            ############## Forward Pass ######################

            losses, generated, grid_for_source, grid_for_prev  = model(data['input'][f],
                                            data['source_frame'][0], generated,
                                            data['grid_source'][f], data['grid'][f],
                                            image = data['target'][f], infer=save_fake)

            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) + loss_dict.get('G_Warp',0)

            # update generator weights
            model.module.optimizer_G.zero_grad()
            loss_G.backward()
            model.module.optimizer_G.step()

            # update discriminator weights
            model.module.optimizer_D.zero_grad()
            loss_D.backward()
            model.module.optimizer_D.step()

            ### display output images
            if save_fake:
                img = (data['source_frame'][0]).cuda()
                warped_source = grid_sample(data['source_frame'][0], data['grid_source'][f].permute(0, 2, 3, 1), padding_mode=opt.grid_padding)
                if not (opt.no_coarse_warp or opt.no_refining_warp):
                    grid = grid_for_source.permute(0, 3, 1, 2).detach()
                    grid = functional.interpolate(grid, (256,256),mode = 'bilinear' )
                    warp = functional.grid_sample(img, grid.permute(0, 2, 3, 1), padding_mode=opt.grid_padding)
                else:
                    warp = warped_source
                visuals = OrderedDict([('synthesized_video_%d'%f, util.tensor2im(generated.data[0])),
                                       ('real_video_%d'%f, util.tensor2im(data['target'][f][0])),
                                       ('warped_source_%d'%f, util.tensor2im(warped_source.data[0])),
                                       ('warped_with_learned_grid_%d'%f, util.tensor2im(warp[0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)
                ############### Backward Pass ####################

        ### linearly decay learning rate after certain iterations
        if lr_decay:
            model.module.update_learning_rate()

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            for key, value in (data['paths']).iteritems():
                print key + "      " + value[0][0]

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()


    #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
