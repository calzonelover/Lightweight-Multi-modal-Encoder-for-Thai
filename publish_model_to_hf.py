import argparse

import torch

from thai_lightweight_encoder.configuration import ThaiLightWeightEncoderConfig
from thai_lightweight_encoder.encoder import ThaiLightWeightEncoderModel
from thai_lightweight_encoder.projector import Projector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_projector_file", default="models/projector_only_distill2.pt")
    parser.add_argument("--target_model_name", default="patomp/thai-light-multimodal-distill")
    args = parser.parse_args()
    
    ThaiLightWeightEncoderConfig.register_for_auto_class()
    ThaiLightWeightEncoderModel.register_for_auto_class("AutoModel")
    config = ThaiLightWeightEncoderConfig()
    encoder = ThaiLightWeightEncoderModel(config)
    
    trained_projector = Projector()
    trained_projector.load_state_dict(torch.load(args.trained_projector_file))
    trained_projector.eval()    
    encoder.projector.load_state_dict(trained_projector.state_dict())
    
    encoder.push_to_hub(args.target_model_name)