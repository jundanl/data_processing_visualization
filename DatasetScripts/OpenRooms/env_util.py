import torch
import torchvision

import image_util


# Convert spherical coordinate to cartesian coordinate
def spherical_to_cartesian(spherical_coords: torch.tensor):
    assert spherical_coords.ndim >= 2 and spherical_coords.shape[0] == 3, \
        "the first dimension of spherical_coords should be 3" # spherical_coords: 3 X N X ...
    # r: radius, theta: polar angle, phi: azimuthal angle
    theta, phi, r = torch.split(spherical_coords, [1, 1, 1], dim=0)
    # Convert spherical coordinates to cartesian coordinates
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.cat([x, y, z], dim=0)


