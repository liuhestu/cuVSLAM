#!/usr/bin/env python3

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

"""Download Tartan Ground dataset for multicamera tracking example."""

import os
import tartanair as ta

data_root = 'dataset/tartan_ground/'
os.makedirs(data_root, exist_ok=True)
ta.init(data_root)
ta.download_ground(
    env=['OldTownFall'],
    version=['anymal'],
    modality=['image'],
    traj=['P2000'],
    camera_name=['lcam_front', 'rcam_front',
                 'lcam_left', 'rcam_left',
                 'lcam_right', 'rcam_right',
                 'lcam_back', 'rcam_back',
                 'lcam_top', 'rcam_top',
                 'lcam_bottom', 'rcam_bottom'],
    unzip=True)
