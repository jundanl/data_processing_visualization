import os
import torch
import torchvision.utils
import time

from openrooms_dataset import *
import scene_util, image_util


def find_index(dataset, key: DataPath):
    for i in range(len(dataset.data_list)):
        if dataset.data_list[i] == key:
            return i


def get_test_data(dataset):
    # Test data
    idx0 = find_index(dataset, DataPath(c='main_xml', s='scene0567_00', idx='8'))
    idx1 = find_index(dataset, DataPath(c='main_xml1', s='scene0014_00', idx='20'))
    idx2 = find_index(dataset, DataPath(c='main_xml', s='scene0014_00', idx='1'))
    idx3 = find_index(dataset, DataPath(c='main_xml1', s='scene0673_05', idx='1'))
    idx4 = find_index(dataset, DataPath(c='mainDiffLight_xml', s='scene0166_00', idx='30'))
    idx5 = find_index(dataset, DataPath(c='main_xml', s='scene0231_00', idx='2'))
    idx6 = find_index(dataset, DataPath(c='main_xml1', s='scene0001_00', idx='1'))
    idx7 = find_index(dataset, DataPath(c='main_xml', s='scene0161_00', idx='1'))
    idx8 = find_index(dataset, DataPath(c='main_xml1', s='scene0014_00', idx='8'))
    idx9 = find_index(dataset, DataPath(c='mainDiffMat_xml', s='scene0129_00', idx='7'))
    test_samples = [idx9, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8]
    return test_samples


def check_is_directional_lighting(dataset, out_dir="./out/test_samples"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    test_samples = get_test_data(dataset)
    for idx in test_samples:
        print(f"\nProcess data idx: {idx}")
        st = time.time()
        or_data = dataset[idx]
        data_time = time.time() - st
        print(f"    Load data time: {data_time} s")
        st = time.time()
        is_DL, (vis, titles) = scene_util.is_directional_lighting(or_data)
        process_time = time.time() - st
        print(f"    Process data time: {process_time} s")
        print(f"    Total time: {data_time + process_time} s")
        print(f"Process idx {idx}, {or_data['img_name']}: {is_DL}")
        plt = image_util.display_images(vis, titles=titles, columns=5, show=False)
        plt.savefig(os.path.join(out_dir, f"{idx}_rs_{is_DL}.jpg"))
        plt.close()
    print("Done!")


or_dataset = OpenRoomsDataset("./data/OpenRooms",
                              "original", "test", False,
                              load_material=False, load_shading=True,
                              load_light_sources=True, load_geometry=True,
                              )
check_is_directional_lighting(or_dataset, "./out/test_samples")
