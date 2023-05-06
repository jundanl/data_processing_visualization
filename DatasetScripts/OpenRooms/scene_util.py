import torch
from torchvision.transforms import Resize, InterpolationMode

import image_util
from image_util import get_tone_mapping_scalar, binary_thresholding, dilate_mask, canny_edge_detection


def estimate_shading_edges(img: torch.tensor, dilate_radius: int = 0, denoise: bool = True):
    # check input shape
    assert img.ndim == 3, "img should be 3D"
    # compute edge map
    # img = img.mean(dim=0, keepdim=True)
    # img = img.clamp(min=0.0, max=1.0)
    img = canny_edge_detection(img, 0.2, 0.4, denoise=denoise)
    # img = binary_thresholding(img, 5, 3)
    # dilate
    if dilate_radius > 0:
        img = dilate_mask(img, dilate_radius)
    return img


def estimate_surface_normal_edges(normal: torch.tensor, dilate_radius: int = 0, denoise: bool = True):
    # Check input shape
    assert normal.ndim == 3 and normal.shape[0] == 3, "normal should be 3D"
    # Convert normal to an image
    # normal = (normal + 1.0) / 2.0

    # Compute edge map using Canny edge detector
    edge_map = canny_edge_detection(normal, 0.2, 0.4, denoise=denoise)

    # dilate
    if dilate_radius > 0:
        edge_map = dilate_mask(edge_map, dilate_radius)
    return edge_map


def estimate_depth_edges(depth: torch.tensor, dilate_radius: int = 0, denoise: bool = True):
    # Check input shape
    assert depth.ndim == 3 and depth.shape[0] == 1, "depth should be 3D"
    # Compute edge map using Canny edge detector
    # depth = (depth - depth.min()) / (depth.max() - depth.min())
    # depth = depth / depth.max()
    edge_map = canny_edge_detection(depth, 0.2, 0.4, denoise=denoise)
    # dilate
    if dilate_radius > 0:
        edge_map = dilate_mask(edge_map, dilate_radius)
    return edge_map


def estimate_geometry_edges(depth: torch.tensor, normal: torch.tensor, dilate_radius: int = 0,
                            denoise: bool = True):
    # Compute edge map using Canny edge detector
    d_edges = estimate_depth_edges(depth, 0, denoise)
    n_edges = estimate_surface_normal_edges(normal, 0, denoise)
    geo_edges = ((d_edges + n_edges) > 0.5).to(torch.float32)
    # dilate
    if dilate_radius > 0:
        geo_edges = dilate_mask(geo_edges, dilate_radius)
    # image_util.display_images([depth, d_edges, normal, n_edges, geo_edges],
    #                           ["depth", "d_edges", "normal", "n_edges", "geo_edges"], columns=5)
    return geo_edges


def is_directional_lighting(or_data):
    # Geometry edges
    depth, normal = or_data["depth"], or_data["normal"]
    geo_edges = estimate_geometry_edges(depth, normal, 5, True)
    resize = Resize(geo_edges.shape[1:], InterpolationMode.BILINEAR, antialias=True)
    # Dilating mask_light
    mask_light = (or_data["mask_light"][0:1] < 0.6).to(torch.float32)
    mask_light_dilated = dilate_mask(mask_light, 5)
    # Final shading edges
    gt_s = or_data["gt_S"]
    gt_s = resize(image_util.denoise_image(gt_s))  # resize and denoise
    tm_scalar = min(get_tone_mapping_scalar(gt_s), 1.0)  # not scale dark images due to the noise
    gt_s *= tm_scalar
    gt_s_edges = estimate_shading_edges(gt_s, 3, denoise=False)
    gt_s_shadow_edges = gt_s_edges * (1.0 - geo_edges) * (1.0 - mask_light_dilated)
    # image_util.display_images([gt_s, gt_s_edges, gt_s_shadow_edges,
    #                            depth, normal, geo_edges,
    #                            mask_light, mask_light_dilated],
    #                           titles=["gt_S", "gt_S_edges", "gt_S_shadow_edges",
    #                                   "depth", "normal", "geo_edges",
    #                                   "mask_light", "mask_light_dilated"],
    #                           columns=3)

    # Categorize final shadow edges caused by either lamps or windows
    assert gt_s_shadow_edges.ndim == 3 and gt_s_shadow_edges.shape[0] == 1, \
        "gt_s_shadow_edges should only have one channel"
    lmp_shadow_edges, wdw_shadow_edges = torch.zeros_like(gt_s_shadow_edges), torch.zeros_like(gt_s_shadow_edges)
    vis, titles = [], []
    lights = or_data["light_sources"]
    assert len(lights) > 0, "No light sources! Please check the data."
    for i in range(len(lights)):
        l = lights[i]
        DS, DSNoOcc = resize(image_util.denoise_image(l.DS)), \
                      resize(image_util.denoise_image(l.DSNoOcc))
        DS, DSNoOcc = DS * tm_scalar, \
                      DSNoOcc * get_tone_mapping_scalar(DSNoOcc)
        edges_DS = estimate_shading_edges(DS, 0, denoise=False)  # shading edges for the direct lighting
        edges_DSNoOcc = estimate_shading_edges(DSNoOcc, 3, denoise=False)  # shading edges for the direct lighting without occlusion
        shading_edges_occ = edges_DS * (1.0 - edges_DSNoOcc)  # shading edges caused by occlusion
        shadow_edges = edges_DS * (1.0 - edges_DSNoOcc) * gt_s_shadow_edges   # hard shadow edges which contribute to final shaidng edges
        if l.is_window:  # shadow edges caused by window lighting
            wdw_shadow_edges += shadow_edges
        else:  # shadow edges caused by lamp lighting
            lmp_shadow_edges += shadow_edges
        # image_util.display_images([DS, edges_DS, DSNoOcc, edges_DSNoOcc, shading_edges_occ, shadow_edges],
        #                           titles=["DS", "edges_DS", "DSNoOcc", "edges_DSNoOcc", "shading_edges_occ",
        #                                   "shadow_edges"],
        #                           columns=6)

    # Check if the scene has strong directional lighting
    lmp_shadow_edges, wdw_shadow_edges = (lmp_shadow_edges > 0.5).to(torch.float32), \
                                         (wdw_shadow_edges > 0.5).to(torch.float32)
    num_lmp_shad, num_wdw_shad = lmp_shadow_edges.sum(), wdw_shadow_edges.sum()
    is_DL = (num_wdw_shad > num_lmp_shad) and (num_wdw_shad > 100)

    # visualization
    # image_util.display_images([or_data["srgb_img"], gt_s, normal, depth, mask_light_dilated,
    #                            gt_s_edges, geo_edges, gt_s_shadow_edges, lmp_shadow_edges, wdw_shadow_edges],
    #                           titles=["srgb_img", "gt_s", "normal", "depth", "mask_light_dilated",
    #                                     "gt_s_edges", "geo_edges", "gt_s_shadow_edges", "lmp_shadow_edges", "wdw_shadow_edges"],
    #                           columns=5)
    vis += [or_data["srgb_img"], gt_s, normal, depth, mask_light_dilated,
            gt_s_edges, geo_edges, gt_s_shadow_edges, lmp_shadow_edges, wdw_shadow_edges]
    titles += [f"srgb_img_{num_lmp_shad}:{num_wdw_shad}_{is_DL}", "gt_s", "normal", "depth", "mask_light_dilated",
               "gt_s_edges", "geo_edges", "gt_s_shadow_edges", "lmp_shadow_edges", "wdw_shadow_edges"]
    return is_DL, (vis, titles)
