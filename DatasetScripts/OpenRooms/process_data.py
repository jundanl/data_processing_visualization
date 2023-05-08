import os
import argparse

import torchvision
from multiprocessing import Pool, Manager

import scene_util, image_util
from openrooms_dataset import OpenRoomsDataset


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


def find_outdoor_lighting_scene(or_dataset_path, split, out_dir, nthread):
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


if __name__ == '__main__':
    # create an argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--func", default="outdoor_lighting", help="function to execute")
    parser.add_argument("--or_dataset_path", default="./data/OpenRooms", help="path to OpenRooms dataset")
    parser.add_argument("--split", default="test", help="split of the dataset")
    parser.add_argument("--out_dir", default="./out", help="output directory")
    parser.add_argument("--nthread", default=8, help="number of threads")

    # parse the arguments
    args = parser.parse_args()

    # assign the arguments to variables
    func = {
        "outdoor_lighting": find_outdoor_lighting_scene,
    }[args.func]
    print("Output directory:", args.out_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print("")

    func(args.or_dataset_path, args.split, args.out_dir, int(args.nthread))