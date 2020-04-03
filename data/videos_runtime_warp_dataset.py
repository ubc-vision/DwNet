import os.path
import torch
from data.base_dataset import BaseDataset, get_params, get_transform, normalize, get_optical_flow_transform
from data.image_folder import make_dataset, atoi, natural_keys, DensePose
from PIL import Image
import numpy as np
import random
import pylab as plt


class Videos_Runtime_Warp_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.input_paths = []
        self.dataset_size = 0
        self.dp = DensePose((opt.loadSize, opt.loadSize), oob_ocluded=True, naive_warp = opt.naive_warp)

        sample_folders = os.listdir(opt.dataroot)
        for folder in sample_folders:
            current_path = os.path.join(opt.dataroot, folder)
            dp_target_folders = (make_dataset(os.path.join(current_path, "dp_target")))
            target_folders = (make_dataset(os.path.join(current_path, "target")))
            dp_target_folders.sort(key=natural_keys)
            target_folders.sort(key=natural_keys)
            for j in range(0, len(target_folders)-2):
                num = range(0,j) + range(j+1, len(target_folders))
                source_index = random.choice(num)
                self.input_paths.append({'dp_target': [dp_target_folders[j]], 'target': [target_folders[j]],
                'source': target_folders[source_index], 'dp_source': dp_target_folders[source_index]})
                for k in range(1, 3):
                    self.input_paths[-1]['dp_target'].append(dp_target_folders[j+k])
                    self.input_paths[-1]['target'].append(target_folders[j+k])

                self.dataset_size += 1


    def __getitem__(self, index):
        transform_img = get_transform(self.opt, {})
        result_dict = {'input': [], 'target': [], 'source_frame': [], 'grid': [], 'grid_source': []}
        output = {}
        output["paths"] = self.input_paths[index]
        current_paths = self.input_paths[index]


        img_source = Image.open(current_paths['source'])
        img_source_tensor = transform_img(img_source.convert('RGB'))


        img_source_dp =  Image.open(current_paths['dp_source'])
        img_source_dp = img_source_dp.convert('RGB')
        np_source_dp = np.array(img_source_dp )

        ### FIRST FRAME OUT OF TWO

        img = Image.open(current_paths['target'][0])
        img_tensor = transform_img(img.convert('RGB'))
        result_dict['target'].append(img_tensor)

        img = Image.open(current_paths['dp_target'][0])
        img_target_dp = img.convert('RGB')
        np_target_dp = np.array(img_target_dp)
        img_tensor = transform_img(img_target_dp)
        result_dict['input'].append(img_tensor)


        grid_source= self.dp.get_grid_warp(np_source_dp, np_target_dp)
        #grid_warp = grid_source[:,:,0]
        #plt.imshow(grid_warp, cmap=plt.cm.gray)
        #plt.show()
        #print grid_source.shape
        grid_source = torch.from_numpy(grid_source).float()
        grid_source = grid_source.permute(2,0,1)

        result_dict['source_frame'].append(img_source_tensor)
        result_dict['grid'].append(grid_source)
        result_dict['grid_source'].append(grid_source)


        #### SECOND FRAME OUT OF TWO

        img_2 = Image.open(current_paths['target'][1])
        img_tensor_2 = transform_img(img_2.convert('RGB'))
        result_dict['target'].append(img_tensor_2)

        img_2 = Image.open(current_paths['dp_target'][1])
        img_target_dp_2 = img_2.convert('RGB')
        np_target_dp_2 = np.array(img_target_dp_2)
        img_tensor_2 = transform_img(img_target_dp_2)
        result_dict['input'].append(img_tensor_2)

        grid_source= self.dp.get_grid_warp(np_source_dp, np_target_dp_2)
        grid_source = torch.from_numpy(grid_source).float()
        grid_source = grid_source.permute(2,0,1)

        grid = self.dp.get_grid_warp(np_target_dp, np_target_dp_2)
        grid = torch.from_numpy(grid).float()
        grid = grid.permute(2,0,1)


        result_dict['source_frame'].append(img_source_tensor)
        result_dict['grid'].append(grid)
        result_dict['grid_source'].append(grid_source)



        #### SECOND FRAME OUT OF TWO

        img_3 = Image.open(current_paths['target'][2])
        img_tensor_3 = transform_img(img_3.convert('RGB'))
        result_dict['target'].append(img_tensor_3)

        img_3 = Image.open(current_paths['dp_target'][2])
        img_target_dp_3 = img_3.convert('RGB')
        np_target_dp_3 = np.array(img_target_dp_3)
        img_tensor_3 = transform_img(img_target_dp_3)
        result_dict['input'].append(img_tensor_3)

        grid_source= self.dp.get_grid_warp(np_source_dp, np_target_dp_3)
        grid_source = torch.from_numpy(grid_source).float()
        grid_source = grid_source.permute(2,0,1)

        grid = self.dp.get_grid_warp(np_target_dp_2, np_target_dp_3)
        grid = torch.from_numpy(grid).float()
        grid = grid.permute(2,0,1)


        result_dict['source_frame'].append(img_source_tensor)
        result_dict['grid'].append(grid)
        result_dict['grid_source'].append(grid_source)



        for key, value in result_dict.iteritems():
            output[key] = torch.stack(value, dim = 0)

        return output

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'Videos_Runtime_Warp_Dataset'
