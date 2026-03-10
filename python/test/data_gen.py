# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

import numpy as np
from scipy.ndimage import zoom
import cuvslam as vslam

def generate_stereo_camera(width: int, height: int, baseline: float = 0.25) -> list[vslam.Camera]:
    """Generate stereo cameras with specified parameters."""
    num_cameras = 2
    cameras = []
    for i in range(num_cameras):
        cam = vslam.Camera(
            size=(width, height),
            focal=(width / 2.0, width / 2.0),
            principal=(width / 2.0, height / 2.0))
        if i == 1:
            cam.rig_from_camera.translation[0] = baseline
        cameras.append(cam)
    return cameras

class ImageGenerator:
    """
    Generate synthetic images for testing.
    The pattern is not ideal and tracking on it may fail, tracking works on each step with 30 steps,
    or on every 2nd step with 60 steps. Tested with 2 cameras 640x480.
    """

    SHIFT = 30            # Shift of pattern in pixels on pre-scaled images between cameras
    SQUARE_SIZE = 40      # Size of squares in checkerboard pattern
    PRESCALE = 3          # By how much original image is bigger than tracked images
    SCALE = 1 / PRESCALE  # Zoom factor to get tracked images from original image

    def __init__(self, cameras: list[vslam.Camera], steps: int = 30) -> None:
        """
        Initialize ImageGenerator with cameras.
        Args:
            cameras: List of cameras.
            steps: Number of steps for generating zoomed images.
        """
        self.cameras = cameras
        self.steps = steps

        W = int(cameras[0].size[0]) # TODO: remove type hint when IntelliSense is fixed
        H = int(cameras[0].size[1])
        h, w = H * ImageGenerator.PRESCALE, W * ImageGenerator.PRESCALE + ImageGenerator.SHIFT
        base_image = np.zeros((h, w), dtype=np.uint8)
        f = float(cameras[0].focal[0])

        # Add vertical offset to the checkerboard pattern to avoid false matches
        square_size = self.SQUARE_SIZE
        for i in range(0, h, 2*square_size):
            for j in range(0, w, 2*square_size):
                col = j // (2*square_size)
                offset = (col * col * 4) % (H // 2)
                base_image[i + offset:i + offset + square_size, j:j + square_size] = 255

        self.cropped = [base_image[:, :-ImageGenerator.SHIFT], base_image[:, ImageGenerator.SHIFT:]]


    def _calc_z_and_scale(self, step) -> tuple[float, float]:
        """
        Calculate the distance to the pattern and scale factor based on the step.
        Returns the distance to the pattern and the scale factor.
        """
        f = float(self.cameras[0].focal[0])
        baseline = float(self.cameras[1].rig_from_camera.translation[0])
        # as rig "moves forward", the base image is zoomed in
        scale = ImageGenerator.SCALE + step / self.steps * 0.5
        # distance to the pattern at the current step
        return f * baseline / (scale * ImageGenerator.SHIFT), scale


    def get_start_distance(self) -> float:
        """
        Get the initial distance to the pattern.
        This is the distance at step 0.
        """
        return self._calc_z_and_scale(0)[0]


    def generate_zoomed_images(self, step: int) -> tuple[list[np.ndarray], float]:
        """
        Generate zoomed images based on the base image and step.
        Returns a list of zoomed images and the distance travelled by cameras
        starting from the initial position (step 0) forward along the z-axis.
        """
        W = int(self.cameras[0].size[0])
        H = int(self.cameras[0].size[1])
        z, scale = self._calc_z_and_scale(step)
        images = []
        for im in self.cropped:
            zoomed = zoom(im, scale)
            start_h = (zoomed.shape[0] - H) // 2
            start_w = (zoomed.shape[1] - W) // 2
            # Crop central region to WxH for tracking
            # Ensure the cropped image is contiguous in memory
            images.append(np.ascontiguousarray(
                zoomed[start_h:start_h + H, start_w:start_w + W]))
        return images, self.get_start_distance() - z
