#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math, time
import torch.nn.functional as F
import diff_gaussian_rasterization_c3
import diff_gaussian_rasterization_c7 
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import sample_camera_rays, get_env_rayd1, get_env_rayd2
import numpy as np

# rayd: x,3, from camera to world points
# normal: x,3
# all normalized
def reflection(rayd, normal):
    refl = rayd - 2*normal*torch.sum(rayd*normal, dim=-1, keepdim=True)
    return refl

def sample_cubemap_color(rays_d, env_map):
    H,W = rays_d.shape[:2]
    outcolor = torch.sigmoid(env_map(rays_d.reshape(-1,3)))
    outcolor = outcolor.reshape(H,W,3).permute(2,0,1)
    return outcolor

def get_refl_color(envmap: torch.Tensor, HWK, R, T, normal_map): #RT W2C
    rays_d = sample_camera_rays(HWK, R, T)
    rays_d = reflection(rays_d, normal_map)
    #rays_d = rays_d.clamp(-1, 1) # avoid numerical error when arccos
    return sample_cubemap_color(rays_d, envmap)

def render_env_map(pc: GaussianModel):
    env_cood1 = sample_cubemap_color(get_env_rayd1(512,1024), pc.get_envmap)
    env_cood2 = sample_cubemap_color(get_env_rayd2(512,1024), pc.get_envmap)
    return {'env_cood1': env_cood1, 'env_cood2': env_cood2}

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, initial_stage = False, more_debug_infos = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    def get_setting(Setting):
        raster_settings = Setting(
            image_height=imH,
            image_width=imW,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        return raster_settings
    
    # init rasterizer with various channels
    Setting_c3 = diff_gaussian_rasterization_c3.GaussianRasterizationSettings
    Setting_c7 = diff_gaussian_rasterization_c7.GaussianRasterizationSettings
    rasterizer_c3 = diff_gaussian_rasterization_c3.GaussianRasterizer(get_setting(Setting_c3))
    rasterizer_c7 = diff_gaussian_rasterization_c7.GaussianRasterizer(get_setting(Setting_c7))

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacities = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    
    bg_map_const = bg_color[:,None,None].cuda().expand(3, imH, imW)
    #bg_map_zero = torch.zeros_like(bg_map_const)

    if initial_stage:
        base_color, _radii = rasterizer_c3(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = None,
            opacities = opacities,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None,
            bg_map = bg_map_const)

        return {
            "render": base_color,
            "viewspace_points": screenspace_points,
            "visibility_filter" : _radii > 0,
            "radii": _radii}

    normals = pc.get_min_axis(viewpoint_camera.camera_center) # x,3
    refl_ratio = pc.get_refl

    input_ts = torch.cat([torch.zeros_like(normals), normals, refl_ratio], dim=-1)
    bg_map = torch.cat([bg_map_const, torch.zeros(4,imH,imW, device='cuda')], dim=0)
    out_ts, _radii = rasterizer_c7(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = input_ts,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None,
        bg_map = bg_map)
    
    base_color = out_ts[:3,...] # 3,H,W
    refl_strength = out_ts[6:7,...] #
    normal_map = out_ts[3:6,...] 

    normal_map = normal_map.permute(1,2,0)
    normal_map = normal_map / (torch.norm(normal_map, dim=-1, keepdim=True)+1e-6)
    refl_color = get_refl_color(pc.get_envmap, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normal_map)
    
    final_image = (1-refl_strength) * base_color + refl_strength * refl_color

    results = {
        "render": final_image,
        "refl_strength_map": refl_strength,
        'normal_map': normal_map.permute(2,0,1),
        "refl_color_map": refl_color,
        "base_color_map": base_color,
        "viewspace_points": screenspace_points,
        "visibility_filter" : _radii > 0,
        "radii": _radii
    }
        
    return results
