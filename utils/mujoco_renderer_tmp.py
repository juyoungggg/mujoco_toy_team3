import mujoco as mj
from mujoco.glfw import glfw
import OpenGL.GL as gl
import numpy as np
import cv2

from utils.mouse_callbacks import MouseCallbacks
from utils.keyboard_callbacks import KeyboardCallbacks

class MuJoCoViewer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Camera 객체 생성
        self.main_cam = mj.MjvCamera()
        self.main_cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.robot_cam = mj.MjvCamera()
        self.robot_cam.type = mj.mjtCamera.mjCAMERA_FIXED
        cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, "camera")
        self.robot_cam.fixedcamid = cam_id

        # 옵션, 씬
        self.opt = mj.MjvOption()
        self.scene_main = mj.MjvScene(self.model, maxgeom=10000)
        self.scene_robot = mj.MjvScene(self.model, maxgeom=10000)

        # GLFW 윈도우 및 context
        glfw.init()
        self.window_main = glfw.create_window(900, 900, "Observer View", None, None)
        self.window_robot = glfw.create_window(640, 480, "Camera View", None, self.window_main)
        glfw.make_context_current(self.window_main)
        self.ctx_main = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        glfw.make_context_current(self.window_robot)
        self.ctx_robot = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        glfw.swap_interval(1)

        mj.mjv_defaultFreeCamera(self.model, self.main_cam)
        mj.mjv_defaultOption(self.opt)

        # 마우스, 키보드 콜백
        mousecallbacks = MouseCallbacks()
        kbdcallbacks = KeyboardCallbacks()
        glfw.set_key_callback(self.window_main, lambda w, k, sc, act, m: kbdcallbacks.keyboardGLFW(w, k, sc, act, m, self.model, self.data, self.opt))        
        glfw.set_cursor_pos_callback(self.window_main, lambda w, x, y: mousecallbacks.mouse_move(w, x, y, self.model, self.scene_main, self.main_cam))
        glfw.set_mouse_button_callback(self.window_main, lambda w, b, act, m: mousecallbacks.mouse_button(w, b, act, m))
        glfw.set_scroll_callback(self.window_main, lambda w, xo, yo: mousecallbacks.scroll(w, xo, yo, self.model, self.scene_main, self.main_cam))

        self.main_cam.azimuth = 0
        self.main_cam.elevation = -50
        self.main_cam.distance = 1
        self.main_cam.lookat = np.array([0.0, 0.0, 0.5])
        self.scene_main.flags[mj.mjtRndFlag.mjRND_SHADOW] = False
        self.scene_main.flags[mj.mjtRndFlag.mjRND_REFLECTION] = False

        self.main_renderer = Renderer(self.window_main, self.scene_main, self.ctx_main, self.main_cam, self.model, self.data, self.opt)
        self.robot_renderer = Renderer(self.window_robot, self.scene_robot, self.ctx_robot, self.robot_cam, self.model, self.data, self.opt)
        
    def step_simulation(self):
        try:
            mj.mj_step(self.model, self.data)
        except Exception as e:
            print(f"시뮬레이션 불안정성 감지: {e}. 데이터를 리셋합니다.")
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            return False  # 에러 발생시 실패 반환 등 추가로 설계 가능
        return True

    def poll_events(self):
        glfw.poll_events()

    # 실제 거리 표시를 위해 extra_info 추가
    def render_main(self, overlay_type="imu", extra_info = None):
        self.main_renderer.render_window(overlay_type=overlay_type, extra_info=extra_info)

    def render_robot(self):
        self.robot_renderer.render_window(overlay_type=None)

    def capture_img(self):
        glfw.make_context_current(self.window_robot)
        w, h = glfw.get_framebuffer_size(self.window_robot)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.float32)
        viewport = mj.MjrRect(0, 0, w, h)
        mj.mjr_readPixels(img, depth, viewport, self.ctx_robot)
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def should_close(self):
        return glfw.window_should_close(self.window_main)

    def terminate(self):
        glfw.terminate()

class Renderer:
    def __init__(self, window, scene, ctx, cam, model, data, opt):
        self.window = window
        self.scene = scene
        self.ctx = ctx
        self.cam = cam
        self.model = model
        self.data = data
        self.opt = opt
        self.w, self.h = glfw.get_framebuffer_size(self.window)

    # 실제 거리 표시를 위해 extra_info 처리 추가
    def render_window(self, overlay_type=None, extra_info=None):
        glfw.make_context_current(self.window)
        viewport = mj.MjrRect(0, 0, self.w, self.h)
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.ctx)
        overlay_text = self.set_overlay_type(overlay_type)

        # 외부 텍스트가 더 있으면 이어 붙임
        if extra_info:
            if overlay_text:
                overlay_text += "\n\n" + extra_info
            else:
                overlay_text = extra_info

        if overlay_text:
            mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL, mj.mjtGridPos.mjGRID_TOPRIGHT, viewport, overlay_text, None, self.ctx)
        glfw.swap_buffers(self.window)
        
    def render_image_for_cv2(self):
        self.render_window() 
        img = gl.glReadPixels(0, 0, self.w, self.h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        img = np.frombuffer(img, dtype=np.uint8).reshape((self.h, self.w, 3))
        img = np.flipud(img)  # OpenGL은 아래가 origin
        return img  # RGB numpy array
        
    def set_overlay_type(self, type):
        if type=="imu":
            overlay = self.get_imu_data()
        elif type=="robot_pose":
            overlay = self.get_robot_pose()
        else:
            overlay = None
        return overlay
        
    def get_imu_data(self):
        qpos = self.data.qpos
        quat_mujoco = qpos[3:7]
        orientation = [quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]]

        accel = self.data.sensordata[0:3]
        gyro  = self.data.sensordata[3:6]

        return (
            "Orientation (Quat)\n"
            f"  x: {orientation[0]:.2f}\n"
            f"  y: {orientation[1]:.2f}\n"
            f"  z: {orientation[2]:.2f}\n"
            f"  w: {orientation[3]:.2f}\n"
            "Gyro \n"
            f"  x: {gyro[0]:.2f}\n"
            f"  y: {gyro[1]:.2f}\n"
            f"  z: {gyro[2]:.2f}\n"
            "Acceleration\n"
            f"  x: {accel[0]:.2f}\n"
            f"  y: {accel[1]:.2f}\n"
            f"  z: {accel[2]:.2f}"
        )
        
    def get_robot_pose(self): 
        base_joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "base_joint")
        qpos_start_addr = self.model.jnt_qposadr[base_joint_id]
        qvel_start_addr = self.model.jnt_dofadr[base_joint_id]
        
        pos = self.data.qpos[qpos_start_addr:qpos_start_addr+3]
        quat_mujoco = self.data.qpos[qpos_start_addr+3:qpos_start_addr+7]
        quat_display = [quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]]
        lin_vel = self.data.qvel[qvel_start_addr:qvel_start_addr+3]
        ang_vel = self.data.qvel[qvel_start_addr+3:qvel_start_addr+6]
        return (
            "Position\n"
            f"  x: {pos[0]:.3f}\n"
            f"  y: {pos[1]:.3f}\n"
            f"  z: {pos[2]:.3f}\n"
            "\n"
            "Orientation (Quat)\n"
            f"  x: {quat_display[0]:.3f}\n"
            f"  y: {quat_display[1]:.3f}\n"
            f"  z: {quat_display[2]:.3f}\n"
            f"  w: {quat_display[3]:.3f}\n"
            "\n"
            "Velocity\n"
            f"  linear x: {lin_vel[0]:.2f}\n"
            f"  angular z: {ang_vel[2]:.2f}"
        )