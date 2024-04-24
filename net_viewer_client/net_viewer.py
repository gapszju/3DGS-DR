import os
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import time, struct

import network as net
import warnings; warnings.filterwarnings("ignore")

class OrbitCamera:
    def __init__(self, img_wh, center, r, rot = None):
        self.W, self.H = img_wh
        self.radius = r
        self.center = center#np.zeros(3)
        if rot is None:
          self.rot = np.eye(3)
        else:
          self.rot = rot

    @property
    def pose(self): # C2W
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot.T
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        #res = qMat#
        return res

    def orbit(self, dx, dy):
        rotvec_x = self.rot[:, 1] * np.radians(0.05 * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                   R.from_rotvec(rotvec_x).as_matrix() @ \
                   self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 1e-4 * self.rot.T @ np.array([dx, dy, dz])

    def dump(self):
        dump_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'cam.npy')
        np.save(dump_path, {'rad': self.radius, 'rot': self.rot, 'cen': self.center}, allow_pickle=True)


class GUI:
    def __init__(self, img_wh, center, radius, rot):
        self.cam = OrbitCamera(img_wh, center, radius, rot)
        self.W, self.H = img_wh
        self.cur_f_idx = 0
        self.auto_play_mode = False
        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)

        # placeholders
        self.dt = 0
        self.mean_samples = 0
        self.img_mode = 0

        self.register_dpg()

    def render_cam(self, cam):
        t = time.time()
        net.send(struct.pack('i', self.img_mode))
        net.send(cam.pose.astype('float32').flatten().tobytes())
        ret = net.read()
        rgb = np.frombuffer(ret, dtype=np.float32).reshape(self.H,self.W,3)
        self.dt = time.time()-t

        return rgb

    def register_dpg(self):
        dpg.create_context()
        dpg.create_viewport(title="net_viewer", width=self.W, height=self.H, resizable=False)

        ## register texture ##
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture")

        ## register window ##
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):
            dpg.add_image("_texture")
        dpg.set_primary_window("_primary_window", True)

        def callback_mode_select(sender, app_data):
            if app_data == 'color': self.img_mode = 0
            elif app_data == 'strength': self.img_mode = 1
            elif app_data == 'base_col': self.img_mode = 2
            elif app_data == 'refl_col': self.img_mode = 3
            elif app_data == 'normal': self.img_mode = 4
            print('Change presentation mode: {}'.format(app_data))

        ## control window ##
        with dpg.window(label="Control", tag="_control_window", width=200, height=150):
            dpg.add_separator()
            dpg.add_text('no data', tag="_log_time")
            dpg.add_radio_button(label = 'mode', items=['color', 'strength', 'base_col', 'refl_col', 'normal'], callback=callback_mode_select)

        ## register camera handler ##
        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            #self.cam.orbit(app_data[1], app_data[2])
            self.cam.orbit(app_data[1], app_data[2])

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.scale(app_data)

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.pan(app_data[1], app_data[2])

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        ## Avoid scroll bar in the window ##
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        ## Launch the gui ##
        dpg.setup_dearpygui()
        dpg.set_viewport_small_icon("assets/icon.png")
        dpg.set_viewport_large_icon("assets/icon.png")
        dpg.show_viewport()

    def render(self):
        while dpg.is_dearpygui_running():
            dpg.set_value("_texture", self.render_cam(self.cam))
            dpg.set_value("_log_time", f'Render time: {1000*self.dt:.2f} ms')
            dpg.render_dearpygui_frame()
        self.cam.dump()


if __name__ == "__main__":
    
    net.init('127.0.0.1', 12357)
    while True:
      print('try to connect server...')
      ret = net.connect()
      if ret: break
      else:
        time.sleep(1)

    info = net.read()
    info = struct.unpack('ii', info)
    img_wh = (info[0], info[1])
    info = net.read()
    info = struct.unpack('ffff', info)
    center = np.array([info[0], info[1], info[2]])
    radius = info[3]
    rot = None
    
    dump_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'cam.npy')
    if os.path.exists(dump_path):
      cam_data = np.load(dump_path, allow_pickle=True).item()
      center = cam_data['cen']
      radius = cam_data['rad']
      rot = cam_data['rot']

    GUI(img_wh, center, radius, rot).render()
    dpg.destroy_context()
