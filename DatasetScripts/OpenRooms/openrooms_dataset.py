import os
import os.path
import time
import random
from collections import namedtuple

import torch, torchvision
import torch.utils.data as data
import numpy as np
from torchvision import io
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as F
from PIL import Image
import cv2

from utils import image_util


DataPath = namedtuple("DataPath", ["c", "s", "idx"])


class OpenRoomsDataset(data.Dataset):
    """
    Load data from OpenRooms dataset
    """
    images_dir = "Image"
    material_dir = "Material"
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
        self.images_dir = os.path.join(self.root, self.images_dir)
        self.material_dir = os.path.join(self.root, self.material_dir)
        self.data_list, self.scene_list = self._get_data_list()

    def __len__(self):
        return len(self.data_list)

    def num_of_scenes(self):
        return len(self.scene_list)

    def _get_data_list(self) -> tuple:
        path = os.path.join(self.root, self.split_files[self.mode])
        flag = self._check_exists([
            self.root,
            self.images_dir,
            self.material_dir,
            path
        ])
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
            # scene_list = scene_list[::5]
        elif self.mode == "val":
            scene_list = scene_list[::50]
        data_list = []
        for s in scene_list:
            for c in self.categories:
                image_scene_path = os.path.join(self.images_dir, c, s)
                if not os.path.exists(image_scene_path):
                    print(f"Not exists image path: {image_scene_path}")
                    continue
                else:
                    material_scene_path = os.path.join(self.material_dir, c.replace("mainDiffLight", "main"), s)
                    if not os.path.exists(material_scene_path):
                        print(f"Not exists material path: {material_scene_path}")
                        continue
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
        GAMMA = 1.0/2.2
        ## input image
        hdr_img_path = os.path.join(self.images_dir, dp.c, dp.s, f"im_{dp.idx}.hdr")
        hdr_img = cv2.imread(hdr_img_path, -1)[:, :, ::-1]
        hdr_img = next(self.numpy_images_2_tensor(hdr_img))
        rgb_img = image_util.tone_mapping(hdr_img, rescale=True, trans2srgb=False)
        srgb_img = image_util.rgb_to_srgb(rgb_img).clamp(min=0.0, max=1.0)
        ## gt_R
        gt_R_path = os.path.join(self.material_dir, dp.c.replace("mainDiffLight", "main"),
                                 dp.s, f"imbaseColor_{dp.idx}.png")
        gt_R = cv2.imread(gt_R_path)[:, :, ::-1].astype(np.float32) / 255.0
        gt_R = next(self.numpy_images_2_tensor(gt_R)) ** (1.0 / GAMMA)
        ## mask
        mask = (gt_R.mean(dim=0, keepdim=True) > 1e-5).to(torch.float32).repeat(3, 1, 1)
        # data augmentation
        if augment_data:
            data_tuple = (srgb_img, rgb_img, gt_R, mask)
            if torch.rand(1) < 0.5:
                data_tuple = (F.hflip(d) for d in data_tuple)
            srgb_img, rgb_img, gt_R, mask = data_tuple
        gt_R = gt_R * mask
        return srgb_img, rgb_img, gt_R, mask, f"{dp.c}_{dp.s}_{dp.idx}"

    def numpy_images_2_tensor(self, *imgs):
        out = (torch.from_numpy(np.transpose(img, (2, 0, 1)).copy()).contiguous().to(torch.float32)
               for img in imgs)
        return out

    def __getitem__(self, index):
        srgb_img, rgb_img, gt_R, mask, filename = self.load_images(self.data_list[index], self.is_train)
        return {"srgb_img": srgb_img,
                "rgb_img": rgb_img,
                "gt_R": gt_R,
                "mask": mask,
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


