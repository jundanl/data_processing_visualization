import io
import os
import math

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
import cv2


def rgb_to_srgb(rgb, gamma=1.0/2.2):
    return rgb.clip(min=0.0) ** gamma


def srgb_to_rgb(srgb, gamma=1.0/2.2):
    return srgb.clip(min=0.0) ** (1.0 / gamma)


def valid_rgI(rgI, v_min=0.0, v_max=1.0):
    assert rgI.ndim == 4 and rgI.size(1) == 3
    assert v_min <= v_max
    ret = torch.ones_like(rgI)
    for i in range(rgI.size(1)):
        c = rgI[:, i:i+1, :, :]
        ret = ret.mul(c >= v_min).mul(c <= v_max)
    s = rgI[:, 0:1, :, :] + rgI[:, 1:2, :, :]
    ret = ret.mul(s >= v_min).mul(s <= v_max)
    return ret


def valid_rgb(rgb, v_min=0.0, v_max=1.0):
    assert rgb.ndim == 4 and rgb.size(1) == 3
    assert v_min <= v_max
    ret = torch.ones_like(rgb)
    for i in range(rgb.size(1)):
        c = rgb[:, i:i+1, :, :]
        ret = ret.mul(c >= v_min).mul(c <= v_max)
    return ret


def rgb_to_chromaticity(rgb):
    """ converts rgb to chromaticity """
    if rgb.ndim == 3:
        sum = torch.sum(rgb, dim=0, keepdim=True).clamp(min=1e-6)
    elif rgb.ndim == 4:
        sum = torch.sum(rgb, dim=1, keepdim=True).clamp(min=1e-6)
    else:
        raise Exception("Only supports image: [C, H, W] or [B, C, H, W]")
    chromat = rgb / sum
    return chromat


def save_srgb_image(image, path, filename):
    # Transform to PILImage
    image_np = np.transpose(image.to(torch.float32).cpu().numpy(), (1, 2, 0)) * 255.0
    image_np = image_np.astype(np.uint8)
    image_pil = Image.fromarray(image_np, mode='RGB')
    # Save Image
    if not os.path.exists(path):
        os.makedirs(path)
    image_pil.save(os.path.join(path,filename))


def get_tone_mapping_scalar(image):
    assert image.ndim == 3
    # MAX_SRGB = 1.077837  # SRGB 1.0 = RGB 1.077837
    src_PERCENTILE = 0.9
    dst_VALUE = 0.8

    vis = image
    if vis.size(0) == 1:
        vis = vis.repeat(3, 1, 1)

    brightness = 0.3 * vis[0, :, :] + 0.59 * vis[1, :, :] + 0.11 * vis[2, :, :]
    src_value = brightness.quantile(src_PERCENTILE)
    if src_value < 1.0e-4:
        scalar = 0.0
    else:
        scalar = math.exp(math.log(dst_VALUE) * 2.2 - math.log(src_value))
    return scalar


def tone_mapping(image, rescale=True, trans2srgb=False):
    assert image.ndim == 3
    vis = image
    if vis.size(0) == 1:
        vis = vis.repeat(3, 1, 1)

    if rescale:
        scalar = get_tone_mapping_scalar(vis)
        vis = scalar * vis

    vis = torch.clamp(vis, min=0)
    if trans2srgb:
        vis = rgb_to_srgb(vis)

    vis = vis.clamp(min=0.0, max=1.0)
    return vis


def adjust_image_for_display(image: torch.tensor, rescale: bool, trans2srgb: bool, src_percentile=0.9999, dst_value=0.85,
                             clip: bool = True):
    assert image.ndim == 3 or image.ndim == 4, "Only supports image: [C, H, W] or [B, C, H, W]"
    vis = image.detach()
    if vis.ndim == 3:
        vis = vis.unsqueeze(0)
    if vis.size(1) == 1:
        vis = vis.repeat(1, 3, 1, 1)

    if rescale:
        src_value = vis.mean(dim=1, keepdim=True)
        src_value = src_value.view(src_value.size(0), -1).quantile(src_percentile, dim=1)
        src_value[src_value < 1e-5] = 1.0
        vis = vis / src_value.view(src_value.size(0), 1, 1, 1) * dst_value
    if trans2srgb:
        vis = rgb_to_srgb(vis)

    if image.ndim == 3:
        vis = vis.squeeze(0)
    return vis.clamp(min=0.0, max=1.0) if clip else vis


def save_intrinsic_images(path, pred_images, label=None, individual=False):
    """ Visualize and save intrinsic images

    :param path: output directory
    :param images: images to be visualized
    :param label: prefix for output files
    :param individual: save visualized images individually or not
    """
    # Visualization for intrinsic images
    ## surface normal
    vis_N = (-pred_images['pred_N']+1) / 2.0

    ## reflectance
    pred_R = pred_images['pred_R']
    vis_R = adjust_image_for_display(pred_R, rescale=False, trans2srgb=True)
    # if pred_R.size(0) == 1:
    #     input_srgb = pred_images['input_srgb']
    #     rgb = srgb_to_rgb(input_srgb)
    #     chromat = rgb_to_chromaticity(rgb)
    #     pred_R = torch.mul(pred_R, chromat)

    ## integrated lighting
    pred_L = pred_images['pred_L']
    L_length = torch.sum(pred_L**2, dim=0, keepdim=True) ** 0.5
    L_direct = pred_L / (L_length + 1e-6)
    vis_L_length = adjust_image_for_display(L_length, rescale=True, trans2srgb=True)
    vis_L_direct = (-L_direct+1)/2.0
    vis_dot_NL = F.cosine_similarity(pred_images['pred_N'], pred_images['pred_L'], dim=0).unsqueeze(0)
    vis_dot_NL = adjust_image_for_display(vis_dot_NL, rescale=False, trans2srgb=True)

    ## shading
    vis_S = adjust_image_for_display(pred_images['pred_S'], rescale=True, trans2srgb=True)

    ## reconstructed image
    vis_rendered_img = adjust_image_for_display(pred_images['rendered_img'], rescale=False, trans2srgb=True)

    ## input image
    vis_srgb = pred_images['input_srgb']

    # Save intrinsic images
    if individual:
        save_srgb_image(vis_N, path, label+'_N.png')
        save_srgb_image(vis_R, path, label+'_R.png')
        save_srgb_image(vis_L_length, path, label+'_L_length.png')
        save_srgb_image(vis_L_direct, path, label+'_L_direct.png')
        save_srgb_image(vis_dot_NL, path, label+'_dotNL.png')
        save_srgb_image(vis_S, path, label+'_S.png')
        save_srgb_image(vis_srgb, path, label+'_rgb.png')
    else:
        # concatenate visualized images into one canva
        vis_merge_h1 = torch.cat((vis_srgb, vis_R, vis_S), 2)
        vis_merge_h2 = torch.cat((vis_L_direct, vis_L_length, vis_dot_NL), 2)
        vis_merge_h3 = torch.cat((vis_N, vis_rendered_img, torch.zeros_like(vis_N)), 2)
        vis_merge = torch.cat((vis_merge_h1, vis_merge_h2, vis_merge_h3), 1)
        save_srgb_image(vis_merge, path, label+'_result.png')


def convert_plot_to_tensor(plot):
    """convert plt to tensor."""
    buf = io.BytesIO()
    plot.savefig(buf, format='jpeg')
    buf.seek(0)
    image = Image.open(buf)
    t = ToTensor()(image)
    return t


def generate_histogram_image(array, title, xlabel, ylabel, bins=100, range=None):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if torch.is_tensor(array):
        array = array.cpu().numpy()
    plt.hist(array, histtype='bar', alpha=0.3, bins=bins, range=range)
    hist_img = convert_plot_to_tensor(plt)
    return hist_img


def numpy_to_tensor(img):
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    else:
        assert False
    return torch.from_numpy(img).contiguous().to(torch.float32)


def tensor_to_numpy(img):
    img = img.cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 4:
        img = np.transpose(img, (0, 2, 3, 1))
    else:
        assert False
    return img


def split_tenors_from_dict(dict, size):
    a, b = {}, {}
    for key, item in dict.items():
        if isinstance(item, torch.Tensor):
            a[key], b[key] = item.split(size, dim=0)
        else:
            a[key] = b[key] = item
    return a, b

