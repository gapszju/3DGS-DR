
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from scene import Scene
from scene.cameras import Camera
import cv2, sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.system_utils import searchForMaxIteration
from gaussian_renderer import render

from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import math
import struct
import time
import gaussian_renderer.network as net
import numpy as np


def render_network(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        pts = gaussians.get_xyz.cpu().numpy()
        
        center = np.mean(pts, axis=0)
        min_pos = np.min(pts, axis=0)
        max_pos = np.max(pts, axis=0)
        radius = np.linalg.norm(max_pos - min_pos)

        test_views = scene.getTestCameras()
        HWK = test_views[0].HWK
        fovX, fovY = test_views[0].FoVx, test_views[0].FoVy
        
        while net.conn is None:
          print('wait for client...')
          net.try_connect()
          time.sleep(1)
          
        info = struct.pack('ii', HWK[1],HWK[0])
        net.send(info)
        info = struct.pack('ffff', *center, radius)
        net.send(info)

        while net.conn is not None:
            try:
                pst_mode = struct.unpack('i', net.read())[0]
                campose = np.frombuffer(net.read(), dtype=np.float32).reshape(4,4)
                campose = np.linalg.inv(campose)

                R = campose[:3,:3]
                T = campose[:3,3:].squeeze(-1)
                view = Camera(0, R, T, fovX, fovY, None, None, None, 0, HWK = HWK)
                res = render(view, gaussians, pipeline, background)
                if pst_mode == 0:
                    image = res["render"]
                elif pst_mode == 1:
                    image = res["refl_strength_map"].expand(3,-1,-1)
                elif pst_mode == 2:
                    image = res["base_color_map"]
                elif pst_mode == 3:  
                    image = res["refl_color_map"]
                elif pst_mode == 4:  
                    image = res["normal_map"]*0.5+0.5
                net.send(torch.clamp(image, min=0, max=1.0).permute(1, 2, 0).flatten().contiguous().cpu().numpy().astype('float32').tobytes())
            except :
                break
            

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    net.init('127.0.0.1', 12357)
    render_network(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    