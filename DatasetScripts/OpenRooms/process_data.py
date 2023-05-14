import os
import argparse
import random
import time

import torch
import torchvision
from multiprocessing import Pool, Manager

import scene_util, image_util
from openrooms_dataset import OpenRoomsDataset


def process_strong_directional_lighting(args):
    # unpack the arguments
    i, dataset, visual_pos_dir, visual_neg_dir, vis_ratio = args
    # get the data
    or_data = dataset[i]
    img_name = or_data["img_name"]
    # check if the image has directional lighting
    is_dl, (vis, titles) = scene_util.is_directional_lighting(or_data)
    if is_dl:  # positive samples
        save_path = os.path.join(visual_pos_dir, f"{img_name.replace('/', '_')}.jpeg")
    else:  # negative samples
        save_path = os.path.join(visual_neg_dir, f"{img_name.replace('/', '_')}.jpeg")
    # visualization
    if random.random() < vis_ratio:
        plt = image_util.display_images(vis, titles=titles, columns=5, show=False)
        plt.savefig(save_path)
        plt.close()
    return i, is_dl, img_name


def process_outdoor_lighting(args):
    # unpack the arguments
    i, dataset, is_outdoor_lighting, lock = args

    # get the input data
    or_data = dataset[i]
    category, scene = or_data["img_name"].split("/")[0:2]

    # check that the scene name is in the is_outdoor_lighting dictionary
    assert f"{category}/{scene}" in is_outdoor_lighting.keys(), \
        f"scene {category}/{scene} not in the is_outdoor_lighting dictionary"

    # if the scene has already been processed, return
    if is_outdoor_lighting[f"{category}/{scene}"]:
        return 0

    # process the scene
    mask_windows = (or_data["mask_light"][0:1] < 0.1).float()
    num_windows_pixels = mask_windows.sum()
    is_bright = ((or_data["rgb_img"].mean(dim=0, keepdim=True) * mask_windows).sum()
                 / num_windows_pixels) > 0.05
    if num_windows_pixels > 20 and is_bright:
        with lock:
            is_outdoor_lighting[f"{category}/{scene}"] = True
    return 1


def find_outdoor_lighting_scenes(or_dataset_path, split, out_dir, nthread, *args):
    assert split in ["train", "test"], f"split {split} not supported"
    print(f"OpenRooms ({split}) dataset path:", or_dataset_path)
    dataset = OpenRoomsDataset(or_dataset_path, "original", split, False,
                               load_material=False, load_shading=False,
                               load_light_sources=False, load_geometry=False,)
    print(f"    number of scenes: {dataset.num_of_scenes()}, number of images: {len(dataset)}")

    print(f"Finding outdoor lighting scenes with {nthread} threads...")
    # create a manager to share the scene_outdoor_lighting dictionary
    manager = Manager()
    is_outdoor_lighting = manager.dict()
    for cs in dataset.scene_list:
        is_outdoor_lighting[cs] = False  # initialize as False
    # process the dataset
    lock = manager.Lock()
    with Pool(processes=nthread) as pool:
        for i, code in enumerate(pool.imap_unordered(process_outdoor_lighting,
                                                     [(i, dataset, is_outdoor_lighting, lock) for i in range(len(dataset))])):
            if i % 200 == 0:
                print(f"Processed the {i}th image: {code}.")
    # save the list of scenes with outdoor lighting
    save_path = os.path.join(out_dir, "outdoor_lighting_split_files", f"{split}.txt")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))  # create the directory if not exists
    cnt = 0
    with open(save_path, "w") as f:
        is_outdoor_lighting = dict(sorted(is_outdoor_lighting.items()))
        for s, flag in is_outdoor_lighting.items():
            if flag:
                f.write(f"{s}\n")
                cnt += 1
    print(f"Done! Saved to {save_path}.")
    print(f"Found {cnt} of {dataset.num_of_scenes()} scenes with outdoor lighting.")


def find_strong_directional_lighting_scenes(or_dataset_path, split, out_dir, nthread, start_idx=0):
    # create the output directory
    out_dir = os.path.join(out_dir, f"strong_directional_lighting_V1/{split}")
    visual_pos_dir = os.path.join(out_dir, "pos")
    visual_neg_dir = os.path.join(out_dir, "neg")
    split_file_dir = os.path.join(out_dir, "strong_directional_lighting_split_files")
    for p in [out_dir, visual_pos_dir, visual_neg_dir, split_file_dir]:
        if not os.path.exists(p):
            os.makedirs(p)
    # create the dataset
    assert split in ["train", "test"], f"split {split} not supported"
    print(f"OpenRooms ({split}) dataset path:", or_dataset_path)
    dataset = OpenRoomsDataset(or_dataset_path, "outdoor_lighting", split, False,
                               load_material=False, load_shading=True,
                               load_light_sources=True, load_geometry=True,)
    print(f"    number of scenes: {dataset.num_of_scenes()}, number of images: {len(dataset)}")

    print(f"Finding strong directional lighting scenes with {nthread} threads...")
    # process the dataset
    file_path = os.path.join(split_file_dir, f"{split}.txt")
    f = open(file_path, "a")
    cnt = 0
    interval = 50
    end = time.time()
    with Pool(processes=nthread) as pool:
        for i, (idx, is_dl, img_name) in enumerate(
                pool.imap_unordered(process_strong_directional_lighting,
                                    [(idx, dataset, visual_pos_dir, visual_neg_dir, 0.1)
                                     for idx in range(start_idx, len(dataset))])):
            if is_dl:
                cnt += 1
                f.write(f"{img_name}\n")
            if (i + 1) % interval == 0:
                print(f"Processed {i+1}/{len(dataset)-start_idx} images.\n"
                      f"    Found {cnt} images with strong directional lighting."
                      f"    Time elapsed: {(time.time()-end)/interval:.2f} seconds/image.")
                end = time.time()

    f.close()
    print(f"Done! Saved to {file_path}.")
    print(f"Found {cnt} of {len(dataset)-start_idx} images with strong directional lighting.")


def show_dataset_size(or_dataset_path, *args):
    print(f"OpenRooms dataset path:", or_dataset_path)
    for split_type in ["original", "outdoor_lighting", "strong_directional_lighting"]:
        print(f"\nChecking {split_type} split files...")
        num_scenes, num_imgs = 0, 0
        for split in ["test", "train"]:
            dataset = OpenRoomsDataset(or_dataset_path, split_type, split, False,
                                       load_material=False, load_shading=False,
                                       load_light_sources=False, load_geometry=False,)
            print(f"    number of {split} scenes: {dataset.num_of_scenes()}, "
                  f"number of {split} images: {len(dataset)}")
            num_scenes += dataset.num_of_scenes()
            num_imgs += len(dataset)
        print(f"    total number of scenes: {num_scenes}, "
              f"total number of images: {num_imgs}")
    print("\nDone!")


if __name__ == '__main__':
    # set the number of threads to 1 to let multiprocessing work properly
    torch.set_num_threads(1)

    # create an argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--func", default="strong_directional", help="function to execute")
    parser.add_argument("--or_dataset_path", default="./data/OpenRooms", help="path to OpenRooms dataset")
    parser.add_argument("--split", default="test", help="split of the dataset")
    parser.add_argument("--out_dir", default="./out", help="output directory")
    parser.add_argument("--nthread", default=4, help="number of threads")
    parser.add_argument("--start_idx", default=0, help="start index of the dataset")

    # parse the arguments
    args = parser.parse_args()

    # assign the arguments to variables
    func = {
        "outdoor_lighting": find_outdoor_lighting_scenes,
        "strong_directional": find_strong_directional_lighting_scenes,
        "show_size": show_dataset_size,
    }[args.func]
    print("Output directory:", args.out_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print("")

    func(args.or_dataset_path, args.split, args.out_dir,
         int(args.nthread), int(args.start_idx))
