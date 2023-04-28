import os
import os.path
import time
# import random
from collections import namedtuple
import glob
import pickle

import torch, torchvision
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
# from torchvision import io
# from torchvision.transforms import RandomResizedCrop
# from torchvision.transforms import functional as F
# from PIL import Image
import cv2

import image_util


DataPath = namedtuple("DataPath", ["c", "s", "idx"])
LightSource = namedtuple("LightSource", ["is_window", "DS", "DSNoOcc", "mask"])


class OpenRoomsDataset(data.Dataset):
    """
    Load data from OpenRooms dataset
    """
    data_dirs = {
        "images": "Image",
        #         "material": "Material",
        "shading": "Shading",
        "shading_direct": "ShadingDirect",
        "mask_light": "Mask",
        "light_source": "LightSource",
        "geometry": "Geometry",
    }
    split_files = {
        "train": "split_files/train.txt",
        "val": "split_files/train.txt",
        "test": "split_files/test.txt",
    }
    categories = ["main_xml", "main_xml1",
                  "mainDiffLight_xml", "mainDiffLight_xml1",
                  "mainDiffMat_xml", "mainDiffMat_xml1"]

    def __init__(self, root: str,
                 mode: str,
                 ) -> None:
        assert mode in ["train", "test", "val"]
        self.mode = mode
        self.is_train = (self.mode in ["train"])

        # check dataset path
        self.root = root
        self.data_dirs = {k: os.path.join(self.root, v) for (k, v) in self.data_dirs.items()}
        self.data_list, self.scene_list = self._get_data_list()

    def __len__(self):
        return len(self.data_list)

    def num_of_scenes(self):
        return len(self.scene_list)

    def _get_data_list(self) -> tuple:
        path = os.path.join(self.root, self.split_files[self.mode])
        flag = self._check_exists([self.root, path] +
                                  [v for (k, v) in self.data_dirs.items()])
        if not flag:
            raise RuntimeError(f"OpenRooms dataset is not found or not complete "
                               f"in the path: {self.root}")
        # load list
        with open(path) as f:
            scene_list = f.readlines()
        scene_list = [s.strip() for s in scene_list]
        scene_list.sort()
        # train/val split
        if self.mode == "train":
            del scene_list[::50]
        elif self.mode == "val":
            scene_list = scene_list[::50]
        data_list = []
        for s in scene_list:
            for c in self.categories:
                image_scene_path = os.path.join(self.data_dirs["images"], c, s)
                if not os.path.exists(image_scene_path):
                    print(f"Not exists image path: {image_scene_path}")
                    continue
                else:
                    pass
                #                     material_scene_path = os.path.join(self.material_dir, c.replace("mainDiffLight", "main"), s)
                #                     if not os.path.exists(material_scene_path):
                #                         print(f"Not exists material path: {material_scene_path}")
                #                         continue
                file_list = os.listdir(image_scene_path)
                file_list.sort()
                for f in file_list:
                    idx = f[len("im_"):-len(".hdr")]
                    assert f == f"im_{idx}.hdr", f"{f}, {f'im_{idx}.hdr'}"
                    data_list.append(DataPath(c, s, idx))
        return data_list, scene_list

    def _check_exists(self, paths) -> bool:
        flag = True
        for p in paths:
            if not os.path.exists(p):
                flag = False
                print(f"Not exists: {p}")
        return flag

    def load_images(self, dp: DataPath, augment_data: bool):
        # load images
        GAMMA = 1.0 / 2.2
        # input image
        hdr_img_path = os.path.join(self.data_dirs["images"], dp.c, dp.s, f"im_{dp.idx}.hdr")
        hdr_img = cv2.imread(hdr_img_path, -1)[:, :, ::-1]
        hdr_img = self.numpy_images_2_tensor(hdr_img)
        tm_scalar = image_util.get_tone_mapping_scalar(hdr_img)
        rgb_img = hdr_img * tm_scalar
        srgb_img = image_util.rgb_to_srgb(rgb_img).clamp(min=0.0, max=1.0)
        # gt_R
        #         gt_R_path = os.path.join(self.data_dirs["material"], dp.c.replace("mainDiffLight", "main"),
        #                                  dp.s, f"imbaseColor_{dp.idx}.png")
        #         gt_R = cv2.imread(gt_R_path)[:, :, ::-1].astype(np.float32) / 255.0
        #         gt_R = next(self.numpy_images_2_tensor(gt_R)) ** (1.0 / GAMMA)
        gt_R = torch.ones_like(hdr_img)
        # gt_S
        gt_S_path = os.path.join(self.data_dirs["shading"], dp.c,
                                 dp.s, f"imshading_{dp.idx}.hdr")
        gt_S = cv2.imread(gt_S_path, -1)[:, :, ::-1]
        gt_S = self.numpy_images_2_tensor(gt_S) * tm_scalar
        # gt_S_direct
        gt_S_direct_path = os.path.join(self.data_dirs["shading_direct"], dp.c.replace("mainDiffMat", "main"),
                                        dp.s, f"imshadingdirect_{dp.idx}.rgbe")
        gt_S_direct = cv2.imread(gt_S_direct_path, -1)[:, :, ::-1]
        gt_S_direct = self.numpy_images_2_tensor(gt_S_direct) * tm_scalar
        # mask_light
        mask_light_path = os.path.join(self.data_dirs["mask_light"], dp.c.replace("mainDiffMat", "main"),
                                       dp.s, f"immask_{dp.idx}.png")
        mask_light = cv2.imread(mask_light_path)[:, :, ::-1].astype(np.float32) / 255.0
        mask_light = (self.numpy_images_2_tensor(mask_light) < 0.6).to(torch.float32)
        # light source information
        lights = []
        light_frame_dir = os.path.join(self.data_dirs['light_source'],
                                       dp.c.replace("mainDiffMat", "main"),
                                       dp.s, f"light_{dp.idx}")
        num_lights = len(glob.glob(os.path.join(light_frame_dir, 'box*.dat')))
        for i in range(num_lights):
            # is window
            with open(os.path.join(light_frame_dir, f"box{i}.dat"), 'rb') as fIn:
                info = pickle.load(fIn)
            is_window = info["isWindow"]
            # light mask
            l_mask = cv2.imread(os.path.join(light_frame_dir, f"mask{i}.png"))[:, :, ::-1].astype(np.float32) / 255.0
            l_mask = self.numpy_images_2_tensor(l_mask)
            # direct shading without occlusion
            l_s_direct_wo_occ = cv2.imread(os.path.join(light_frame_dir, f"imDSNoOcclu{i}.rgbe"), -1)[:, :, ::-1]
            l_s_direct_wo_occ = self.numpy_images_2_tensor(l_s_direct_wo_occ) * tm_scalar
            # direct shading
            l_s_direct = cv2.imread(os.path.join(light_frame_dir, f"imDS{i}.rgbe"), -1)[:, :, ::-1]
            l_s_direct = self.numpy_images_2_tensor(l_s_direct) * tm_scalar
            #
            lights.append(LightSource(is_window, l_s_direct, l_s_direct_wo_occ, l_mask))

        # surface normal
        normal_path = os.path.join(self.data_dirs["geometry"], dp.c.replace("mainDiffLight", "main"), dp.s,
                                   f"imnormal_{dp.idx}.png")
        normal = cv2.imread(normal_path)[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
        normal = self.numpy_images_2_tensor(normal)
        normal = F.normalize(normal, dim=0, p=2)

        # mask
        mask = (gt_R.mean(dim=0, keepdim=True) > 1e-5).to(torch.float32) * (1.0 - mask_light)

        # data augmentation
        if augment_data:
            pass
        #             data_tuple = (hdr_img, srgb_img, rgb_img, gt_R, gt_S, gt_S_direct, mask, mask_light)
        #             if torch.rand(1) < 0.5:
        #                 data_tuple = (F.hflip(d) for d in data_tuple)
        #             hdr_img, srgb_img, rgb_img, gt_R, gt_S, gt_S_direct, mask, mask_light = data_tuple
        gt_R = gt_R * mask
        return hdr_img, srgb_img, rgb_img, gt_R, gt_S, gt_S_direct, \
               mask, mask_light, lights, normal, f"{dp.c}_{dp.s}_{dp.idx}"

    def numpy_images_2_tensor(self, *imgs):
        if len(imgs) == 1:
            out = torch.from_numpy(np.transpose(imgs[0], (2, 0, 1)).copy()).contiguous().to(torch.float32)
        else:
            out = (torch.from_numpy(np.transpose(img, (2, 0, 1)).copy()).contiguous().to(torch.float32)
                   for img in imgs)
        return out

    def __getitem__(self, index):
        hdr_img, srgb_img, rgb_img, gt_R, gt_S, gt_S_direct, \
        mask, mask_light, lights, normal, filename = self.load_images(self.data_list[index], self.is_train)
        return {"hdr_img": hdr_img,
                "srgb_img": srgb_img,
                "rgb_img": rgb_img,
                "gt_R": gt_R,
                "gt_S": gt_S,
                "gt_S_direct": gt_S_direct,
                "mask": mask,
                "mask_light": mask_light,
                "light_sources": lights,
                "normal": normal,
                "index": index,
                "img_name": filename,
                "dataset": "OpenRooms"}


def check_dataset_split(dataset_openrooms, batch_size, num_workers, disp_iters, visualize_dir=None) -> None:
    if visualize_dir is not None:
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)

    data_loader_openrooms = torch.utils.data.DataLoader(dataset_openrooms,
                                                        shuffle=False, drop_last=False,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers)
    end = time.time()
    for index, data_openrooms in enumerate(data_loader_openrooms):
        sample = data_openrooms["srgb_img"]
        if (index + 1) % disp_iters == 0:
            print(f"index: {index}, image shape: {sample.shape}, "
                  f"time/batch: {(time.time()-end)/disp_iters}")
            end = time.time()
        if visualize_dir is not None and ((index+1) % disp_iters == 0):
            for b in range(sample.size(0)):
                vis_imgs = [v[b] for k, v in data_openrooms.items()
                            if k not in ["index", "dataset", "img_name"]]
                torchvision.utils.save_image(vis_imgs,
                                             os.path.join(visualize_dir, f"{data_openrooms['img_name'][b]}.jpeg"))


def is_OpenRooms_complete(dataset_dir: str, batch_size=8, num_workers=1, visualize_dir=None) -> bool:
    if visualize_dir is not None:
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)
        train_sample_dir = os.path.join(visualize_dir, "train_samples")
        val_sample_dir = os.path.join(visualize_dir, "val_samples")
        test_sample_dir = os.path.join(visualize_dir, "test_samples")
    else:
        train_sample_dir = val_sample_dir = test_sample_dir = None

    dataset_openrooms_train, dataset_openrooms_val, dataset_openrooms_test = get_train_val_test_sets(dataset_dir)
    print(f"OpenRooms _ train:\n"
          f"\tscenes: {dataset_openrooms_train.num_of_scenes()}, size: {len(dataset_openrooms_train)}, batch_size: {batch_size}")
    check_dataset_split(dataset_openrooms_train, batch_size, num_workers, 1000, train_sample_dir)

    print(f"OpenRooms _ val:\n"
          f"\tscenes: {dataset_openrooms_val.num_of_scenes()}, size: {len(dataset_openrooms_val)}, batch_size: {batch_size}")
    check_dataset_split(dataset_openrooms_val, batch_size, num_workers, 50, val_sample_dir)

    print(f"OpenRooms _ test:\n"
          f"\tscenes: {dataset_openrooms_test.num_of_scenes()}, size: {len(dataset_openrooms_test)}, batch_size: {batch_size}")
    check_dataset_split(dataset_openrooms_test, batch_size, num_workers, 50, test_sample_dir)
    return True


def get_train_val_test_sets(dataset_dir: str):
    return OpenRoomsDataset(dataset_dir, "train"), \
           OpenRoomsDataset(dataset_dir, "val"), \
           OpenRoomsDataset(dataset_dir, "test")


