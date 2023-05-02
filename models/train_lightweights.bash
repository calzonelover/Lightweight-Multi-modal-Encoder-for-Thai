#!/bin/bash

python finetune_lightweight_encoder.py --target_save_file projector_only_distill.pt --is_only_distill 1

python finetune_lightweight_encoder.py --target_save_file projector_zero_alpha.pt --alpha 0.0

python finetune_lightweight_encoder.py --target_save_file projector_high_alpha.pt --alpha 200.0