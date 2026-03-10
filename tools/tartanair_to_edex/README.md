Dataset converter
===================
----------
This converter is aimed to provide the ability to convert TartanAir dataset into Kitti format.

#### Examples
> python3 -m dataset_converter --seq_path <path to folder with unpacked sequences>

This will:
1) rename the images in the format that reporter demands
2) produce 'edex' folder with edexes
3) produce 'gt' folder with gts.
#### Installation

> pip install .
