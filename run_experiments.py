# run_experiments.py
import os
effnet_settings = [
    {"depth_mult": 1.0, "width_mult": 1.0, "resolution": 224},
    {"depth_mult": 1.0, "width_mult": 1.1, "resolution": 240},
    {"depth_mult": 1.2, "width_mult": 1.4, "resolution": 260},
    # 추가 조합들...
]

for i, setting in enumerate(effnet_settings):
    os.system(f"python train.py --model_name efficientnet_b0 --d {setting['depth_mult']} --w {setting['width_mult']} --r {setting['resolution']} --run_name effnet_exp_{i}")
