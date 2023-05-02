import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_DATASETS_CACHE"] = "/workspace/cache"

from datasets import load_dataset

import clip
from multilingual_clip import pt_multilingual_clip
import transformers
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="patomp/thai-mscoco-2014-captions")
    parser.add_argument("--model_name", default="M-CLIP/XLM-Roberta-Large-Vit-B-32")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--target_path", default="teacher_encode.val.hf")
    args = parser.parse_args()
    
    text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(args.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    visual_model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = dataset.map(
        lambda x: {
            "embed_image": visual_model.encode_image(preprocess(x["image"]).unsqueeze(0).to(device))[0].to("cpu").detach().numpy().astype(float)
        },
        batched=False
    )    
    dataset = dataset.map(
        lambda x: {
            'embed_th_sentence': text_model.forward(
                [x["th_sentences_raw"][0],],
                tokenizer
            )[0].detach().numpy().astype(float)
        },
        batched=False
    )
    dataset.save_to_disk(args.target_path)