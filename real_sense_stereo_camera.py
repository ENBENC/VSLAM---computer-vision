import pyrealsense2 as rs2
import numpy as np
from enum import Enum, IntEnum, unique, auto
from common_lab_utils import (Size, StereoPair)


@unique
class CameraIndex(IntEnum):
    LEFT = 1
    RIGHT = 2


@unique
class LaserMode(Enum):
    ON = auto()
    OFF = auto()


class RealSenseStereoCamera:

    def __init__(self):
        self.ir_size = Size(width=640, height=480)

        self._pipe = rs2.pipeline()

        cfg = rs2.config()
        cfg.disable_all_streams()
        mode = rs2.format.y8
        cfg.enable_stream(rs2.stream.infrared, CameraIndex.LEFT, self.ir_size.width, self.ir_size.height, mode)
        cfg.enable_stream(rs2.stream.infrared, CameraIndex.RIGHT, self.ir_size.width, self.ir_size.height, mode)
        cfg.enable_stream(rs2.stream.color, self.ir_size.width, self.ir_size.height, rs2.format.bgr8, 30)
        self._pipe.start(cfg)

        self.set_laser_mode(LaserMode.OFF)

    def __del__(self):
        self._pipe.stop()

    def __str__(self):
        device = self._pipe.get_active_profile().get_device()
        serial_number = device.get_info(rs2.camera_info.serial_number)
        device_product_line = str(device.get_info(rs2.camera_info.product_line))
        
        return ("RealSense:\n"
            f"  product line: {device_product_line}\n"
            f"  serial: {serial_number}\n"
            f"  resolution: {self.get_resolution(CameraIndex.LEFT)}\n"
            f"  framerate: {self.get_framerate(CameraIndex.LEFT)}"
        )

    def get_stereo_pair(self) -> StereoPair:
        data = self._pipe.wait_for_frames()
        frame_l = np.asanyarray(data.get_infrared_frame(CameraIndex.LEFT).get_data())
        frame_r = np.asanyarray(data.get_infrared_frame(CameraIndex.RIGHT).get_data())
        return StereoPair(frame_l, frame_r)
    
    def get_stereo_pair_rgb(self) -> StereoPair:
        data = self._pipe.wait_for_frames()
        frame_l = np.asanyarray(data.get_color_frame().get_data())
        return StereoPair(frame_l, frame_l)

    def get_framerate(self, camera: CameraIndex):
        profile = self.get_video_stream_profile(camera)
        return profile.fps()

    def get_resolution(self, camera: CameraIndex) -> Size:
        profile = self.get_video_stream_profile(camera)
        return Size(width=profile.width(), height=profile.height())

    def get_calibration_matrix(self, camera: CameraIndex):
        i = self.get_video_stream_profile(camera).get_intrinsics()
        return np.array([
            [i.fx, 0, i.ppx],
            [0, i.fy, i.ppy],
            [0, 0, 1]
        ])

    def get_distortion(self, camera: CameraIndex):
        d = self.get_video_stream_profile(camera).get_intrinsics().coeffs
        return np.array([d])

    def get_pose(self):
        selection = self._pipe.get_active_profile()
        left_stream = selection.get_stream(rs2.stream.infrared, CameraIndex.LEFT)
        right_stream = selection.get_stream(rs2.stream.infrared, CameraIndex.RIGHT)
        e = left_stream.get_extrinsics_to(right_stream)
        R = np.asarray(e.rotation).reshape((3, 3))
        t = np.asarray(e.translation)

        return R, t

    def get_video_stream_profile(self, camera: CameraIndex):
        return self._pipe.get_active_profile().get_stream(rs2.stream.infrared, camera).as_video_stream_profile()

    def set_laser_mode(self, laser_mode: LaserMode):
        """'True' or anyting"""
        depth_sensor = self._pipe.get_active_profile().get_device().first_depth_sensor()

        if not depth_sensor.supports(rs2.option.emitter_enabled):
            return

        if laser_mode is LaserMode.ON:
            depth_sensor.set_option(rs2.option.emitter_enabled, 1)
        else:
            depth_sensor.set_option(rs2.option.emitter_enabled, 0)
