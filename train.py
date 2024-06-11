#!/usr/bin/env python3

from arg_parse import train_arg
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from model.helper import generate_square_subsequent_mask
from model.helper import create_mask
from model.ParameterShareTransformer import ParameterShareTransformer
from tqdm import tqdm 
from torch.cuda.amp import autocast, GradScaler
from model.optim import InverseSqrtScheduler, LabelSmoothingLoss
import os 
import sentencepiece as spm
from data.dataloader import get_dataloader
from torch.utils.data import DataLoader
from model.ParameterShareTransformer import Encoder, Decoder, ParameterShareTransformer

from accelerate import Accelerator

def main(args):
    
    #  helpful for memory limitation, train on larger batch sizes by accumulating the gradients over multiple batches before updating the weights.
    accelerator = Accelerator(gradient_accumulation_steps=2, mixed_precision="fp16")
    data_dir = args.data

    # build a dataset and dataloader 
    train_loader = get_dataloader(data_dir, 'en', 'de', 'train', args.batch_size)
    print(f"Loaded Training dataset")

    # INIT 
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # for debugging 
    # device = "cpu"
    device = accelerator.device
    model = ParameterShareTransformer(args.input_dim, args.output_dim, args.src_pad_idx, args.tgt_pad_idx, args.max_tokens, device)


    # criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=eval(args.adam_betas), weight_decay=args.weight_decay)
    # criterion = LabelSmoothingLoss(ignore_index=args.tgt_pad_idx, smoothing=args.label_smoothing).to(device)
    criterion = LabelSmoothingLoss(smoothing=args.label_smoothing).to(device)
    scheduler = InverseSqrtScheduler(optimizer, args.warmup_updates, args.warmup_init_lr, args.lr)
    scaler = GradScaler()
    if accelerator.
    print(f"Total number of parameters of the model: {sum(p.numel() for p in model.parameters())}")

    train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)
    # Training 
    model.train()
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            with accelerator.accumulate(model):
            
                src_batch, tgt_batch = batch # (seq_length, bs)
                # print(f"src_batch: {src_batch.shape}")

                src_batch = pad_sequence(src_batch, batch_first=False, padding_value=args.src_pad_idx).to(device)
                tgt_batch = pad_sequence(tgt_batch, batch_first=False, padding_value=args.tgt_pad_idx).to(device)
                # print(f"src_batch after pad_sequence: {src_batch.shape}")
                
                # Truncate sequences that exceed the max_token length
                src_batch = src_batch[:, :args.max_tokens] # (bs, seq_length)
                tgt_batch = tgt_batch[:, :args.max_tokens]
                # print(f"src_batch after trucating up to max_tokens: {src_batch.shape}")
                
                src_batch = src_batch.reshape(-1, args.batch_size)
                tgt_batch = tgt_batch.reshape(-1, args.batch_size) # (seq_lengh, bs)

                # print(f"src_batch after reshaping into (seq_len, bs): {src_batch.shape}")
                # the target sequence is usually shifted by one position during training. This is because the model needs to predict the next token given all the previous tokens
                tgt_input = tgt_batch[:-1, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_mask = create_mask(src_batch, tgt_input, args.tgt_pad_idx)

                optimizer.zero_grad()
                # with autocast():
                with accelerator.autocast():
                    # batch_first = True for encoder and decoder 
                    src_batch = src_batch.reshape(args.batch_size, -1)
                    tgt_input = tgt_input.reshape(args.batch_size, -1)
                    
                    output = model(src_batch, tgt_input, src_mask, tgt_mask, memory_mask, src_padding_mask, tgt_padding_mask)
                    tgt_out = tgt_batch[1:, :]
                    loss = criterion(output.view(-1, output.shape[-1]), tgt_out.view(-1))
                
                # loss.backward()
                # scaler.scale(loss).backward()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), args.clip_norm)
                # if args.clip_norm > 0:
                #     scaler.unscale_(optimizer)
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

                optimizer.step()
                # scaler.step(optimizer)
                # scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                
                # progress_bar.set_postfix({"Loss": loss.item(), "LR": optimizer.param_groups[0]["lr"]})
            progress_bar.set_postfix({"Loss": loss.item(), "LR": scheduler.step()})

        print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader)}")

if __name__ == '__main__':
    args = train_arg()
    main(args)