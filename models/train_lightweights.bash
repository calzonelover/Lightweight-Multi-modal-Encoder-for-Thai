#!/bin/bash

python finetune_lightweight_encoder.py --target_save_file projector_only_distill2.pt --is_only_distill 1 --n_epochs 200 --learning_rate 0.1

python finetune_lightweight_encoder.py --target_save_file projector_high_alpha2.pt --alpha 400.0 --n_epochs 200 --learning_rate 0.1

python finetune_lightweight_encoder.py --target_save_file projector_zero_alpha2.pt --alpha 0.0 --n_epochs 200 --learning_rate 0.1