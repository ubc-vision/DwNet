### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import numpy as np

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.batchSize = 1

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)

    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

video_group = 0

for i, data in enumerate(dataset):
    with torch.no_grad():
        if i >= opt.how_many:
            break

        data["dp_target"] = data["dp_target"].permute(1, 0, 2, 3, 4)
        data["grid"] = data["grid"].permute(1, 0, 2, 3, 4)
        data["grid_source"] = data["grid_source"].permute(1, 0, 2, 3, 4)

        generated = model.inference(data['dp_target'][0],
                                        data['source_frame'], data['source_frame'],
                                        data['grid_source'][0], data['grid_source'][0])


        img_path = data['path'][0]
        frame_number = str(0)
        print generated.size()
        print('process image... %s' % img_path+"   "+str(0))
        visualizer.save_images(webpage, util.tensor2im(generated.squeeze(dim = 0)), img_path, frame_number)
        visuals = OrderedDict([('synthesized_image', util.tensor2im(generated.squeeze(dim = 0)))])
        visualizer.display_current_results(visuals, 100, 12345)


        for i in range(1, data["dp_target"].shape[0]):
            if opt.prev_frame_num  == 0:
                generated = model.inference(data['dp_target'][i],
                                                data['source_frame'], data['source_frame'],
                                                data['grid_source'][i], data['grid_source'][i])
            else:
                generated = model.inference(data['dp_target'][i],
                                                data['source_frame'], generated,
                                                data['grid_source'][i], data['grid'][i-1])

            img_path = data['path'][0]
            frame_number = str(i)
            print('process image... %s' % img_path + "   " + str(i))
            visualizer.save_images(webpage,util.tensor2im(generated.squeeze(dim = 0)), img_path, frame_number)
            visuals = OrderedDict([('synthesized_image', util.tensor2im(generated.squeeze(dim = 0)))])
            visualizer.display_current_results(visuals, 100, 12345)

webpage.save()
