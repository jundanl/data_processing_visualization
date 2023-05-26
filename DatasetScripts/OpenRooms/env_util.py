import torch
import torchvision

import image_util


def spherical_to_cartesian(spherical_coords: torch.tensor):
    """
    Convert spherical coordinates to cartesian coordinates.
    Local cartesian and spherical coordinate systems:
        theta: clockwise angle from Z axis,
        phi: clockwise angle from X axis
          Z
          ^        X
          |theta/
          |   /
          | /
          |--------> Y
        /  \ phi
      /     \
    /        \
    :param spherical_coords: 3 X N X ...
    :return: cartesian_coords: 3 X N X ...
    """
    assert spherical_coords.ndim >= 2 and spherical_coords.shape[0] == 3, \
        "the first dimension of spherical_coords should be 3"  # spherical_coords: 3 X N X ...
    # r: radius, theta: polar angle, phi: azimuthal angle
    theta, phi, r = torch.split(spherical_coords, [1, 1, 1], dim=0)
    # Convert spherical coordinates to cartesian coordinates
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.cat([x, y, z], dim=0)


def spherical_gaussian_parameters_2_cartesian(sgs, coord_transform=None, normal=None):
    """
    Convert the mean vectors of spherical gaussian to the cartesian coordinate system.
    :param sgs:  N X 6 [X H X W], where N is the number of spherical gaussian lobes
    :param coord_transform: a function that transforms the SG to another coordinate system
    :param normal: the surface normal vector(s)/map
    :return: sg_cart_mean: N X 3 [X H X W], sg_lamb: N X 1 [X H X W], sg_weight: N X 3 [X H X W]
    """
    assert sgs.shape[1] == 6, \
        "sgs should have 6 channels: theta, phi, lamb, weight"  # N X 6 [X H X W]
    theta, phi, lamb, weight = torch.split(sgs, [1, 1, 1, 3], dim=1)  # N X C [X H X W]
    sg_sph_mean = torch.cat([theta, phi,
                             torch.ones_like(theta)],
                            dim=1)  # N X 3 [X H X W]
    swap_dims = list(range(sgs.ndim))
    swap_dims[0], swap_dims[1] = 1, 0
    sg_cart_mean = spherical_to_cartesian(sg_sph_mean.permute(*swap_dims))  # 3 X N [X H X W]
    sg_cart_mean = sg_cart_mean.permute(*swap_dims)  # N X 3 [X H X W]
    if coord_transform is not None:  # transform the coordinates to the world/camera system
        assert normal is not None, "normal should be provided when coord_transform is not None"
        if sg_cart_mean.ndim == 2:  # N X 3
            assert normal.ndim == 1 and normal.shape[0] == 3, \
                "normal should be a 3D vector when sg_cart_mean is a 2D tensor"
            sg_cart_mean = sg_cart_mean[:, :, None, None]
            normal = normal[:, None, None]
        assert sg_cart_mean.ndim == 4 and normal.ndim == 3 and \
               sg_cart_mean.shape[2:] == normal.shape[1:], \
               "sg_cart_mean and normal should have the same height and width"
        # resize = torchvision.transforms.Resize(sgs.shape[2:4], antialias=True)
        # normal = resize(normal)
        # normal = normal / torch.linalg.vector_norm(normal, ord=2, dim=0, keepdim=True).clamp(min=1e-6)
        sg_cart_mean = coord_transform(sg_cart_mean, normal)  # N X 3 X H X W
        if sg_sph_mean.ndim == 2:
            sg_cart_mean = sg_cart_mean.squeeze(2).squeeze(2)  # N X 3
    return sg_cart_mean, lamb, weight


def panorama_spherical_coords(height: int, width: int, theta_max: float, phi_max: float):
    """
    Create spherical coordinates for a panorama:
    generates a grid of spherical coordinates (theta, phi) for each pixel in the panorama image.
    Local cartesian and spherical coordinate systems:
    (Note that the axis orientations are different from the camera coordinate system)
        theta: clockwise angle from Z axis,
        phi: clockwise angle from X axis
          Z
          ^        X
          |theta/
          |   /
          | /
          |--------> Y
        /  \ phi
      /     \
    /        \

    :param height: the height of the panorama image
    :param width: the width of the panorama image
    :param theta_max: the maximum polar angle
    :param phi_max: the maximum azimuthal angle
    :return: spherical_coords: 3 X height X width, spherical coordinates of the panorama image
    """
    # Create a grid of spherical coordinates
    theta_range = ((torch.arange(height, dtype=torch.float32) + 0.5) / height) * theta_max
    phi_range = ((torch.arange(width, dtype=torch.float32) + 0.5) / width - 0.5) * phi_max
    theta, phi = torch.meshgrid(theta_range, phi_range, indexing="ij")
    # Convert grid coordinates to spherical coordinates
    # y_coords, x_coords = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width),
    #                                     indexing="ij")
    # theta = y_coords * theta_max
    # phi = x_coords * phi_max

    # Concatenate theta and phi to create the spherical coordinates tensor
    spherical_coords = torch.stack([theta, phi, torch.ones_like(theta)], dim=0)
    return spherical_coords  # 3 X height X width


def render_spherical_gaussian_env_maps(sgs: torch.tensor, height: int = 180, width: int = 360,
                                       coord_transform=None, normal=None, visualize=False):
    """
    Render spherical gaussian environment maps at one location.
    :param sgs: N X 6, where N is the number of spherical gaussian lobes
    :param height: the height of the rendered environment map
    :param width: the width of the rendered environment map
    :param coord_transform: a function that transforms the SG to another coordinate system
    :param normal: the surface normal vector of the surface
    :param visualize: whether to return the visualized rendered environment maps
    :return:    env_sum: 3 X height X width, final rendered environment map
                env_maps: N X 3 X height X width, environment maps of each spherical gaussian lobe
                # (vis, titles): the visualized rendered environment maps and their titles
    """
    # Check the input shape
    assert sgs.ndim == 2 and sgs.shape[1] == 6, \
        "sgs should be: N X 6"  # N: number of spherical gaussian lobes

    # Compute a grid of cartesian coordinates (X, Y, Z) for each pixel in the panorama.
    pano_sph_coords = panorama_spherical_coords(height, width,
                                                torch.pi, 2 * torch.pi)  # 3 X H X W
    _pano_cart_coords = spherical_to_cartesian(pano_sph_coords) # 3 X H X W

    # Reorient the axes to conform to the camera coordinate system: [X, Y, Z] -> [Y, Z, -X]
    pano_cart_coords = torch.zeros_like(_pano_cart_coords)
    pano_cart_coords[0, :, :] = _pano_cart_coords[1, :, :]
    pano_cart_coords[1, :, :] = _pano_cart_coords[2, :, :]
    pano_cart_coords[2, :, :] = -_pano_cart_coords[0, :, :]
    pano_cart_coords = pano_cart_coords[None]  # 1 X 3 X H X W
    # print("Verifying the spherical coordinates to cartesian coordinates conversion:")
    # veri_x = [0, width//4,  width//2,  width*3//4,  0,         width//2,  width-1,       0, width-1]
    # veri_y = [0, height//3, height//3, height//3,   height//2, height//2, height//2,   height-1, height-1]
    # for i in range(len(veri_x)):
    #     x = veri_x[i]
    #     y = veri_y[i]
    #     print(f"Img ({y}, {x}): {pano_sph_coords[:-1, y, x]/torch.pi*180} => {pano_cart_coords[0, :, y, x]}")

    # Compute the cartesian coordinates of the spherical gaussian parameters
    sg_cart_mean, sg_lamb, sg_weight = spherical_gaussian_parameters_2_cartesian(sgs, coord_transform, normal)
    sg_cart_mean = sg_cart_mean[:, :, None, None]  # 12 X 3 X 1 X 1
    sg_lamb = sg_lamb[:, :, None, None]  # 12 X 1 X 1 X 1
    sg_weight = sg_weight[:, :, None, None]  # 12 X 3 X 1 X 1
    # for i in range(sg_cart_mean.shape[0]):
    #     print(f"SG {i}:")
    #     print(f"  mean: {sg_cart_mean[i, :, 0, 0]}")
    #     print(f"  lamb: {sg_lamb[i, :, 0, 0].item():.2f}")
    #     print(f"  weight: {sg_weight[i, :, 0, 0].mean().item():.2f}")

    # Render the SG environment maps
    dot_product = torch.sum(pano_cart_coords * sg_cart_mean, dim=1, keepdim=True)  # 12 X 1 X height X width
    e = torch.exp(-sg_lamb * (1 - dot_product))  # 12 X 1 X height X width
    envs = e * sg_weight  # 12 X 3 X height X width
    env_sum = envs.sum(dim=0)  # 3 X height X width
    # print(sg_theta.shape, sg_phi.shape, sg_lamb.shape, sg_weight.shape, envs.shape, env_sum.shape)

    # Visualize the environment maps
    if visualize:
        vis = [(pano_cart_coords[0] + 1) / 2.0]  # 3 X H X W
        titles = ["Pano cartesian coords in camera"]
        vis += [envs[i]/envs[i].max() for i in range(envs.shape[0])]
        for i in range(envs.shape[0]):
            # Format each element of mean and weight to 2 decimal places
            mean = [f'{m:.2f}' for m in sg_cart_mean[i, :, 0, 0].tolist()]
            # weight = [f'{w:.2f}' for w in sg_weight[i, :, 0, 0].tolist()]
            titles += [f"SG {i}:\n"
                       f"  mean: {mean}\n"
                       f"  lamb: {sg_lamb[i, 0, 0, 0]:.2f}\n"
                       f"  weight: {sg_weight[i, :, 0, 0].mean():.2f}"]
        vis += [env_sum/env_sum.max()]
        titles += ["Sum env"]
        image_util.display_images(vis, titles, figsize_base=5, columns=3, show=True)
    return env_sum, envs


def estimate_pw_dominant_lighting_direction(or_data, coord_transform=None, visualize=False):
    """
    Estimate the pixel-wise dominant lighting directions from the SG environment representation.
        Camera Coordinate System:
          Y
          ^
          |
          |
          |
          |--------> X
        /
      /
     Z
    :param or_data: data from the OpenRooms dataset
    :param coord_transform: a function to transform the coordinates to the world/camera system
    :param visualize: whether to visualize the results
    :return: direct_of_major_light: the pixel-wise dominant lighting directions, 3 X height X width
            mask_light: the mask indicating the pixels lit by directional lighting, 1 X height X width
    """
    # Get data
    pw_sgs = or_data["SG_env"]
    assert pw_sgs.ndim == 4 and pw_sgs.shape[1] == 6, \
        f"pw_sgs should be 4D tensor with shape (numSGs, 6, height, width), but got {pw_sgs.shape}"
    mask_light = (or_data["mask_light"][0:1] < 0.6).to(torch.float32)
    resize = torchvision.transforms.Resize(pw_sgs.shape[2:4], antialias=True)
    mask_light = resize(mask_light)
    mask_light = (mask_light > 0.01).to(torch.float32)

    # Spherical gaussian parameters
    normal = resize(or_data["normal"])
    normal = normal / torch.linalg.vector_norm(normal, ord=2, dim=0, keepdim=True).clamp(min=1e-6)
    sg_cart_mean, lamb, weight = \
        spherical_gaussian_parameters_2_cartesian(pw_sgs, coord_transform=coord_transform, normal=normal)

    # Find the dominant lighting direction according to the weight
    weight_mean = weight.mean(dim=1, keepdim=True)  # 12 X 1 X H X W
    max_w, max_idx = torch.max(weight_mean, dim=0, keepdim=True)  # 1 X 1 X H X W
    max_idx = max_idx.repeat(1, sg_cart_mean.shape[1], 1, 1)  # 1 X 3 X H X W
    direct_of_major_light = torch.gather(sg_cart_mean, dim=0, index=max_idx).squeeze(0)  # 3 X H X W
    lamb_of_major_light = torch.gather(lamb, dim=0, index=max_idx[:, 0:1, :, :]).squeeze(0)  # 1 X H X W
    # debug gathering
    # _midx = max_idx[0, 0, :, :]
    # _H, _W = _midx.shape
    # s = torch.zeros(3, _H, _W)
    # for y in range(_H):
    #     for x in range(_W):
    #         s[:, y, x] = sg_cart_mean[_midx[y, x], :, y, x]
    # print(major_light_direct.shape, s.shape)
    # assert torch.eq(s, major_light_direct).all(), f"major_light_direct should be the same as s"

    # Label highly-directional lighting areas
    mask_strong_directional = 1.0 - mask_light  # Exclude light sources
    mask_strong_directional *= lamb_of_major_light > 200.0  # Narrow spread
    # cos(10 degrees) = 0.9848
    is_dissimilar_light = (direct_of_major_light[None] * sg_cart_mean).sum(dim=1, keepdim=True) < 0.9848  # 12 X 1 X H X W
    is_strong_light = max_w < weight_mean * 10.0  # 12 X 1 X H X W
    num_other_competitive_lights = (is_dissimilar_light * is_strong_light).to(torch.float32).sum(dim=0)  # 1 X H X W
    mask_strong_directional = mask_strong_directional * (num_other_competitive_lights < 0.1)  # 1 X H X W

    # Visualize
    if visualize:
        # visualize pixel-wise major lighting direction
        vis_light_direct = (direct_of_major_light + 1.0) / 2.0
        vis_normal = (normal + 1.0) / 2.0
        vis = [or_data["srgb_img"], vis_normal, or_data["gt_S"],
               vis_light_direct, mask_strong_directional, vis_light_direct * mask_strong_directional]
        titles = ["srgb_img", "normal", "gt_S",
                  "direct_of_major_light", "mask_strong_directional", "strong_directional_light"]
        # select the brightest pixel
        gt_s = or_data["gt_S"]  # C X H X W
        resize_as_s = torchvision.transforms.Resize(gt_s.shape[1:3], antialias=True)
        intensity = gt_s.mean(dim=0, keepdim=True)
        area_avg_intensity = torch.nn.functional.avg_pool2d(intensity.unsqueeze(0), 5, stride=1, padding=2).squeeze(0)
        area_avg_intensity *= mask_strong_directional
        assert area_avg_intensity.ndim == 3 and area_avg_intensity.shape[0] == 1
        v, max_idx = torch.max(area_avg_intensity.view(-1), dim=0)
        y, x = divmod(max_idx.item(), area_avg_intensity.shape[2])
        circled_input = image_util.draw_a_circle(resize_as_s(or_data["srgb_img"]), (x, y),
                                                 radius=3, color=(255, 0, 0), thickness=2)
        scaled_area_avg = area_avg_intensity / area_avg_intensity.max()
        scaled_area_avg = image_util.draw_a_circle(scaled_area_avg, (x, y),
                                                   radius=3, color=(1, 0, 0), thickness=2)
        l_direct_at_xy = direct_of_major_light[:, y, x].tolist()
        l_direct_at_xy = [f'{v:.2f}' for v in l_direct_at_xy]
        print(f"Major lighting directiong at {(x, y)}: {l_direct_at_xy}")
        # print(f"value {v}: {area_avg_intensity[0, y, x]}. ({x, y})")
        # print(f"s: {scaled_area_avg[:, y, x]}")
        vis += [circled_input, area_avg_intensity, scaled_area_avg]
        titles += ["circled_input", "area_avg_intensity", f"scaled_area_avg\n{l_direct_at_xy}"]
        image_util.display_images(vis, titles, columns=5, show=True)
        # visualize the environment maps of the brightest pixel
        _, _ = render_spherical_gaussian_env_maps(
            pw_sgs[:, :, y, x],
            coord_transform=coord_transform,
            normal=normal[:, y, x],
            visualize=True)
        # image_util.display_images(vis + vis_envs, titles + vis_env_titles, columns=5, show=True)
    return direct_of_major_light, mask_strong_directional
