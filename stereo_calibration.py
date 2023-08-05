import cv2
import numpy as np
from common_lab_utils import (Size, StereoPair)
#from kitti_interface import (KittiCamera)
from real_sense_stereo_camera import (RealSenseStereoCamera, CameraIndex)
from pylie import SO3, SE3


class StereoCalibration:
    def __init__(self, k_left, k_right, d_left, d_right, R, t, img_size) -> None:
        self._img_size = img_size

        # Intrinsics
        self._k_left = k_left
        self._k_right = k_right
        self._d_left = d_left
        self._d_right = d_right
        # Extrinsics
        self._R = R
        self._t = t

        self._compute_rectification_mapping()

    def __str__(self):
        return (
            f"\nK.left:\n{self._k_left}\n"
            f"\nd.left:\n{self._d_left}\n"
            f"\nK.right:\n{self._k_right}\n"
            f"\nd.right:\n{self._d_right}\n"
            f"\nR:\n{self._R}\n"
            f"\nt:\n{self._t}\n"
        )

    @classmethod
    def from_file(cls, intrinsic_filename, extrinsic_filename, img_size: Size):
        """
        Load stereo calibration parameters from files.
        
        :param intrinsic_filename: *.yml file containing intrinsics.
        :param extrinsic_filename: *.yml file containing extrinsics.
        :param img_size: The image size for which the calibration is valid.
        """
        pass

    @classmethod
    def from_realsense(cls, stereo_camera: RealSenseStereoCamera):
        """Load stereo calibration parameters from a RealSense camera."""
        k_left = stereo_camera.get_calibration_matrix(CameraIndex.LEFT)
        k_right = stereo_camera.get_calibration_matrix(CameraIndex.RIGHT)

        d_left = stereo_camera.get_distortion(CameraIndex.LEFT)
        d_right = stereo_camera.get_distortion(CameraIndex.RIGHT)

        r, t = stereo_camera.get_pose()
        img_size = stereo_camera.get_resolution(CameraIndex.LEFT)

        return cls(k_left, k_right, d_left, d_right, r, t, img_size)
    
    #@classmethod
    #def from_kitti(cls, kitti_cam: KittiCamera):
    #    """Load stereo calibration parameters from a KittiCamera."""
    #    cam_l = kitti_cam.calibration['GrayLeft']
    #    cam_r = kitti_cam.calibration['GrayRight']

    #    k_left = cam_l["calibration"]
    #    k_right = cam_r["calibration"]

    #    d_left = cam_l["distortion"]
    #    d_right = cam_r["distortion"]

    #    r = cam_r["rotation"]
    #    t = cam_r["translation"]
    #    img_size = Size(cam_l["size"][0], cam_l["size"][1])

    #    return cls(k_left, k_right, d_left, d_right, r, t, img_size)

    @property
    def baseline(self):
        """The distance between the origins of the cameras"""
        return 1.0 / self._q[3, 2]
    
    @property
    def p_left(self):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self._R
        T[:3, 3] = self._t
        return self._k_left@T[:3]
    
    @property
    def p_right(self):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self._R
        T[:3, 3] = self._t
        return self._k_right@T[:3]

    @property
    def f(self):
        """The f-coefficient of the Q-matrix"""
        return self._q[2, 3]

    @property
    def img_size(self):
        """The image size for which the calibration is valid. (as (width, height))"""
        return self._img_size.as_cv_size

    @property
    def q(self):
        """The computed Q-matrix from the calibration parameters."""
        return self._q

    @property
    def k_left(self):
        """The intrinsic parameters of the left camera"""
        return self._k_left

    @property
    def k_right(self):
        """The intrinsic parameters of the right camera."""
        return self._k_right

    @property
    def distortion_left(self):
        """The distortion parameters of the left camera"""
        return self._d_left

    @property
    def distortion_right(self):
        """The distortion parameters of the right camera."""
        return self._d_right

    @property
    def pose_left_right(self):
        """The rotation and translation of the right camera relative to the left camera."""
        return SE3((SO3(self._R), self._t))

    def _compute_rectification_mapping(self):
        img_size = self.img_size

        r_left, r_right, p_left, p_right, self._q, valid1, valid2 = cv2.stereoRectify(
            self._k_left, self._d_left,
            self._k_right, self._d_right,
            img_size,
            self._R, self._t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            newImageSize=img_size
        )

        self._map_left_x, self._map_left_y = cv2.initUndistortRectifyMap(self._k_left, self._d_left, r_left, p_left,
                                                                         img_size, cv2.CV_16SC2)
        self._map_right_x, self._map_right_y = cv2.initUndistortRectifyMap(self._k_right, self._d_right, r_right,
                                                                           p_right, img_size, cv2.CV_16SC2)

    def rectify(self, raw_stereo_pair: StereoPair):
        """
        Rectify a stereo pair with the loaded calibration parameters.
        
        :param raw_stereo_pair: The input images
        :return: rectified images, ready for stereo processing
        """
        if raw_stereo_pair.left.dtype == np.uint16:
            raw_stereo_pair.left = np.uint8(raw_stereo_pair.left * 255.0 / 65535.0)
            raw_stereo_pair.right = np.uint8(raw_stereo_pair.right * 255.0 / 65535.0)

        rectified_left = cv2.remap(raw_stereo_pair.left, self._map_left_x, self._map_left_y, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(raw_stereo_pair.right, self._map_right_x, self._map_right_y, cv2.INTER_LINEAR)

        return StereoPair(rectified_left, rectified_right)
