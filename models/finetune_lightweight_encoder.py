import argparse
import os
os.environ["HF_DATASETS_CACHE"] = "/workspace/cache"

from datasets import load_from_disk
from pythainlp import word_vector, word_tokenize
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import transformers

from projector import Projector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_save_file")
    parser.add_argument("--train_dataset_hf_path", default="teacher_encode.train.hf")
    parser.add_argument("--val_dataset_hf_path", default="teacher_encode.val.hf")
    parser.add_argument("--target_path", default="teacher_encode.val.hf")
    parser.add_argument("--alpha", default=200.0)
    parser.add_argument("--learning_rate", default=0.01)
    parser.add_argument("--batch_size", default=1024)
    parser.add_argument("--n_epochs", default=80)
    parser.add_argument("--is_only_distill", default=0)
    args = parser.parse_args()
    
    print(args)
    
    alpha = float(args.alpha)
    learning_rate = float(args.learning_rate)
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs)
    is_only_distill = int(args.is_only_distill)
    
    model = word_vector.WordVector(model_name="thai2fit_wv")
    projector = Projector(input_embedding_dim=300)
    
    def embed_sentence(examples):
        chunks = []
        for text in examples["text"]:
            embed = model.sentence_vectorizer(text, use_mean=True)[0]
            chunks.append(embed.astype(float))
        return {"embed_sentence": chunks}
    
    def prep_dataset(_dataset):
        _dataset = _dataset.map(lambda x: {"text": x["th_sentences_raw"][0]})
        _dataset = _dataset.map(embed_sentence, batched=True)
        _dataset = _dataset.with_format(type="torch", columns=["embed_image", "embed_th_sentence", "embed_sentence"])
        return _dataset
    
    dataset = load_from_disk(args.train_dataset_hf_path)
    dataset = prep_dataset(dataset)   
    print(dataset)
    print(dataset[0])
    val_dataset = load_from_disk(args.val_dataset_hf_path)
    val_dataset = prep_dataset(val_dataset)
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    projector.train()
    distil_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = Adam(projector.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_losses = {"tot": [], "ce": [], "distil": []}
    for epoch in range(1, n_epochs+1): 
        # training loop in an epoch
        tot_losses, ce_losses, diltil_losses = [], [], []
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            image_z = batch["embed_image"]
            raw_z = batch["embed_sentence"]
            teacher_z = batch["embed_th_sentence"]
            student_z = projector(raw_z)
            bs = image_z.size()[0]

            distil_loss_val = distil_loss(student_z, teacher_z)
            image_z = image_z / image_z.norm(dim=1, keepdim=True)
            student_z = student_z / student_z.norm(dim=1, keepdim=True)
            logits = torch.einsum("ij,kl->ik", image_z, student_z) * torch.exp(projector.temperature)
            labels = torch.eye(bs)
            ce_loss_val = 0.5 * (ce_loss(logits, labels) + ce_loss(logits.T, labels))
            total_loss_val = distil_loss_val if is_only_distill else ce_loss_val + alpha * distil_loss_val

            total_loss_val.backward()
            optimizer.step()
            
            ce_losses.append(ce_loss_val.item())
            diltil_losses.append(distil_loss_val.item())
            tot_losses.append(total_loss_val.item())

        # compute loss on the validation for regularization
        val_loss_val = []
        with torch.no_grad():
            for batch in val_dataloader:
                image_z = batch["embed_image"]
                raw_z = batch["embed_sentence"]
                teacher_z = batch["embed_th_sentence"]
                student_z = projector(raw_z)
                bs = image_z.size()[0]

                distil_loss_val = distil_loss(student_z, teacher_z)     
                image_z = image_z / image_z.norm(dim=1, keepdim=True)
                student_z = student_z / student_z.norm(dim=1, keepdim=True)
                logits = torch.einsum("ij,kl->ik", image_z, student_z) * torch.exp(projector.temperature)
                labels = torch.eye(bs)
                ce_loss_val = 0.5 * (ce_loss(logits, labels) + ce_loss(logits.T, labels))
                total_loss_val = distil_loss_val if is_only_distill else ce_loss_val + alpha * distil_loss_val
                val_loss_val.append(total_loss_val.item())

        val_loss_val = sum(val_loss_val) / float(len(val_loss_val))
        scheduler.step(val_loss_val)
            
        ep_ce_los = sum(ce_losses) / float(len(ce_losses))
        ep_distil_los = sum(diltil_losses) / float(len(diltil_losses))
        ep_tot_los = sum(tot_losses) / float(len(tot_losses))
        print(epoch, ep_tot_los, ep_ce_los, ep_distil_los)
    
    torch.save(projector.state_dict(), args.target_save_file)