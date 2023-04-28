import torch
from torchvision.transforms import Resize, InterpolationMode

from image_util import get_tone_mapping_scalar, binary_thresholding, dilate_mask, canny_edge_detection


def estimate_shading_edges(img: torch.tensor, dilate_radius: int = 0):
    # check input shape
    assert img.ndim == 3, "img should be 3D"
    # compute edge map
    img = canny_edge_detection(img, 50, 200, True)
    # img = binary_thresholding(img, 5, 3)
    # dilate
    if dilate_radius > 0:
        img = dilate_mask(img, dilate_radius)
    return img


def estimate_surface_normal_edges(normal: torch.tensor, dilate_radius: int = 0):
    # Check input shape
    assert normal.ndim == 3 and normal.shape[0] == 3, "normal should be 3D"
    # Convert normal to an image
    normal = (normal + 1.0) / 2.0
    # Compute edge map using Canny edge detector
    edge_map = torch.zeros_like(normal[0:1])
    for i in range(normal.shape[0]):
        edge_map += canny_edge_detection(normal[i:i+1], 50, 200, False)
    edge_map = (edge_map > 0.5).to(torch.float32)
    # dilate
    if dilate_radius > 0:
        edge_map = dilate_mask(edge_map, dilate_radius)
    return edge_map


def is_directional_lighting(or_data):
    assert or_data["gt_S"].ndim == 3, "gt_S should be 3D"
    # normal edges
    normal = or_data["normal"]
    normal = Resize(or_data["gt_S"].shape[1:], InterpolationMode.BILINEAR, antialias=True)(normal)
    normal = normal / ((normal ** 2).sum(dim=0, keepdim=True) ** 0.5).clamp(min=1e-6)
    normal_edges = estimate_surface_normal_edges(normal, 3)
    # dilated mask_light
    mask_light = or_data["mask_light"][0:1]
    mask_light = (Resize(or_data["gt_S"].shape[1:], InterpolationMode.BILINEAR, antialias=True)(mask_light) > 0.5).to(torch.float32)
    mask_light_dilated = dilate_mask(mask_light, 3)
    # shading edges
    gt_s = or_data["gt_S"]
    tm_scalar = get_tone_mapping_scalar(gt_s)
    gt_s = gt_s * tm_scalar
    gt_s_edges = estimate_shading_edges(gt_s, 3) * (1.0 - normal_edges) * (1.0 - mask_light_dilated)
    # computed shading edges caused by either lamps or windows
    lights = or_data["light_sources"]
    lmp_shadow_edges, wdw_shadow_edges = torch.zeros_like(gt_s), torch.zeros_like(gt_s)
    vis = []
    for i in range(len(lights)):
        l = lights[i]
        edges_DS = estimate_shading_edges(l.DS * tm_scalar)
        edges_DSNoOcc = estimate_shading_edges(l.DSNoOcc * get_tone_mapping_scalar(l.DSNoOcc))
        shadow_edges = edges_DS * gt_s_edges
        if l.is_window:
            wdw_shadow_edges += shadow_edges
        else:
            lmp_shadow_edges += shadow_edges
        vis += [l.DS * tm_scalar, edges_DS,
                l.DSNoOcc * get_tone_mapping_scalar(l.DSNoOcc), edges_DSNoOcc,
                shadow_edges]
    # check if the scene has strong directional lighting
    lmp_shadow_edges, wdw_shadow_edges = (lmp_shadow_edges > 0.5).to(torch.float32), \
                                         (wdw_shadow_edges > 0.5).to(torch.float32)
    num_lmp_shadow_pixels = lmp_shadow_edges.sum() / lmp_shadow_edges.shape[0]
    num_wdw_shadow_pixels = wdw_shadow_edges.sum() / wdw_shadow_edges.shape[0]
    is_DL = (num_wdw_shadow_pixels > num_lmp_shadow_pixels) and (num_wdw_shadow_pixels > 50)
    # visualization
    vis += [or_data["srgb_img"], gt_s, mask_light_dilated, gt_s_edges, normal_edges, lmp_shadow_edges, wdw_shadow_edges]
    return is_DL, vis
