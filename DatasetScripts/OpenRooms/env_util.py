import torch
import torchvision

import image_util


# Convert spherical coordinate to cartesian coordinate
def spherical_to_cartesian(spherical_coords: torch.tensor):
    assert spherical_coords.ndim >= 2 and spherical_coords.shape[0] == 3, \
        "the first dimension of spherical_coords should be 3"  # spherical_coords: 3 X N X ...
    # r: radius, theta: polar angle, phi: azimuthal angle
    theta, phi, r = torch.split(spherical_coords, [1, 1, 1], dim=0)
    # Convert spherical coordinates to cartesian coordinates
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.cat([x, y, z], dim=0)


# Estimate the pixel-wise dominant lighting directions from the environment maps
def estimate_pw_dominant_lighting_direction(or_data, coord_transform=None):
    # Get data
    pw_sgs = or_data["SG_env"]
    assert pw_sgs.ndim == 4 and pw_sgs.shape[1] == 6, \
        f"pw_sgs should be 4D tensor with shape (numSGs, 6, height, width), but got {pw_sgs.shape}"
    mask_light = (or_data["mask_light"][0:1] < 0.6).to(torch.float32)
    resize = torchvision.transforms.Resize(pw_sgs.shape[2:4], antialias=True)
    mask_light = resize(mask_light)
    mask_light = (mask_light > 0.1).to(torch.float32)

    # Spherical gaussian parameters
    theta, phi, lamb, weight = torch.split(pw_sgs, [1, 1, 1, 3], dim=1)  # 12 X C X H X W
    sg_sph_coords = torch.cat([theta, phi,
                               torch.ones_like(theta)],
                              dim=1)  # 12 X 3 X H X W
    sg_cart_coords = spherical_to_cartesian(sg_sph_coords.permute(1, 0, 2, 3))  # 3 X 12 X H X W
    sg_cart_coords = sg_cart_coords.permute(1, 0, 2, 3)  # 12 X 3 X H X W
    if coord_transform is not None:  # transform the coordinates to the world/camera system
        normal = or_data["normal"]
        normal = resize(normal)
        normal = normal / torch.linalg.norm(normal, ord=2, dim=0, keepdim=True).clamp(min=1e-6)
        sg_cart_coords = coord_transform(sg_cart_coords, normal)  # 12 X 3 X H X W

    # Find the dominant lighting direction according to the weight
    weight_mean = weight.mean(dim=1, keepdim=True)  # 12 X 1 X H X W
    max_w, max_idx = torch.max(weight_mean, dim=0, keepdim=True)  # 1 X 1 X H X W
    max_idx = max_idx.repeat(1, sg_cart_coords.shape[1], 1, 1)  # 1 X 3 X H X W
    major_light_direct = torch.gather(sg_cart_coords, dim=0, index=max_idx).squeeze(0)  # 3 X H X W
    # debug gathering
    # _midx = max_idx[0, 0, :, :]
    # _H, _W = _midx.shape
    # s = torch.zeros(3, _H, _W)
    # for y in range(_H):
    #     for x in range(_W):
    #         s[:, y, x] = sg_cart_coords[_midx[y, x], :, y, x]
    # print(major_light_direct.shape, s.shape)
    # assert torch.eq(s, major_light_direct).all(), f"major_light_direct should be the same as s"

    # Exclude non-directional lighting areas
    mask_strong_directional = 1.0 - mask_light
    dissimilar_direct = (major_light_direct[None] * sg_cart_coords).sum(dim=1, keepdim=True) < 0.95  # 12 X 1 X H X W
    is_strong_light = max_w < weight_mean * 10.0  # 12 X 1 X H X W
    num_other_strong_light = (dissimilar_direct * is_strong_light).to(torch.float32).sum(dim=0)  # 1 X H X W
    mask_strong_directional = mask_strong_directional * (num_other_strong_light < 0.1)  # 1 X H X W
    return major_light_direct, mask_strong_directional
