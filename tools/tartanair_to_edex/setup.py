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

from setuptools import setup, find_packages

setup(
    name='dataset-converter',
    version='0.1',
    description='Helps to convert datasets to Kitti format',
    author='NVIDIA CORPORATION',

    packages=find_packages(),
    package_data={
        '': ['cfg/*.json']
    },

    install_requires=[
        'click == 7.1.2',
        'numpy == 1.26.4',
        'scipy == 1.14.1',
    ],
    python_requires='>=3.6.0'
)
