
import torch
from scene import Scene
import os, time
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_env_map
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import get_lpips_model

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, save_ims):
    if save_ims:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        #gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        color_path = os.path.join(render_path, 'rgb')
        normal_path = os.path.join(render_path, 'normal')
        makedirs(color_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)
        #makedirs(gts_path, exist_ok=True)

    LPIPS = get_lpips_model(net_type='vgg').cuda()
    ssims = []
    psnrs = []
    lpipss = []
    render_times = []

    if save_ims: # save env light
        ltres = render_env_map(gaussians)
        torchvision.utils.save_image(ltres['env_cood1'], os.path.join(model_path, 'light1.png'))
        torchvision.utils.save_image(ltres['env_cood2'], os.path.join(model_path, 'light2.png'))
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.refl_mask = None # when evaluating, refl mask is banned
        t1 = time.time()
        rendering = render(view, gaussians, pipeline, background)
        render_time = time.time() - t1
        
        render_color = rendering["render"][None]
        gt = view.original_image[None, 0:3, :, :]

        ssims.append(ssim(render_color, gt).item())
        psnrs.append(psnr(render_color, gt).item())
        lpipss.append(LPIPS(render_color, gt).item())
        render_times.append(render_time)

        if save_ims:
            normal_map = rendering['normal_map'] * 0.5 + 0.5
            torchvision.utils.save_image(render_color, os.path.join(color_path, '{0:05d}.png'.format(idx)))
            torchvision.utils.save_image(normal_map, os.path.join(normal_path, '{0:05d}.png'.format(idx)))
    
    ssim_v = np.array(ssims).mean()
    psnr_v = np.array(psnrs).mean()
    lpip_v = np.array(lpipss).mean()
    fps = 1.0/np.array(render_times).mean()
    print('psnr:{},ssim:{},lpips:{},fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))
    dump_path = os.path.join(model_path, 'metric.txt')
    with open(dump_path, 'w') as f:
        f.write('psnr:{},ssim:{},lpips:{},fps:{}'.format(psnr_v, ssim_v, lpip_v, fps))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, save_ims : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, save_ims)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.save_images)