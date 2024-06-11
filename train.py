#!/usr/bin/env python3

from arg_parse import train_arg
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from model.helper import generate_square_subsequent_mask, create_mask
from model.ParameterShareTransformer import ParameterShareTransformer
from tqdm import tqdm
from model.optim import InverseSqrtScheduler, LabelSmoothingLoss
import os
import sentencepiece as spm
from data.dataloader import get_dataloader
from torch.utils.data import DataLoader
from accelerate import Accelerator

def main(args):
    
    # Helpful for memory limitation, train on larger batch sizes by accumulating the gradients over multiple batches before updating the weights.
    accelerator = Accelerator(gradient_accumulation_steps=2, mixed_precision="fp16")
    data_dir = args.data

    # Build a dataset and dataloader 
    train_loader = get_dataloader(data_dir, 'en', 'de', 'train', args.batch_size)
    if accelerator.is_main_process:
        print(f"Loaded Training dataset")

    # Initialize device
    device = accelerator.device

    # Initialize model
    model = ParameterShareTransformer(args.input_dim, args.output_dim, args.src_pad_idx, args.tgt_pad_idx, args.max_tokens, device)

    # Initialize optimizer, criterion, scheduler, and scaler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=eval(args.adam_betas), weight_decay=args.weight_decay)
    criterion = LabelSmoothingLoss(smoothing=args.label_smoothing).to(device)
    scheduler = InverseSqrtScheduler(optimizer, args.warmup_updates, args.warmup_init_lr, args.lr)
    
    if accelerator.is_main_process:
        print(f"Total number of parameters of the model: {sum(p.numel() for p in model.parameters())}")

    # Prepare dataloader, model, and optimizer with accelerator
    train_loader, model, optimizer = accelerator.prepare(train_loader, model, optimizer)

    # Training
    model.train()
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            with accelerator.accumulate(model):
                src_batch, tgt_batch = batch

                src_batch = pad_sequence(src_batch, batch_first=False, padding_value=args.src_pad_idx).to(device)
                tgt_batch = pad_sequence(tgt_batch, batch_first=False, padding_value=args.tgt_pad_idx).to(device)
                
                # Truncate sequences that exceed the max_token length
                src_batch = src_batch[:, :args.max_tokens]
                tgt_batch = tgt_batch[:, :args.max_tokens]
                
                src_batch = src_batch.reshape(-1, args.batch_size)
                tgt_batch = tgt_batch.reshape(-1, args.batch_size)

                tgt_input = tgt_batch[:-1, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_mask = create_mask(src_batch, tgt_input, args.tgt_pad_idx)

                optimizer.zero_grad()
                with accelerator.autocast():
                    src_batch = src_batch.reshape(args.batch_size, -1)
                    tgt_input = tgt_input.reshape(args.batch_size, -1)
                    
                    output = model(src_batch, tgt_input, src_mask, tgt_mask, memory_mask, src_padding_mask, tgt_padding_mask)
                    tgt_out = tgt_batch[1:, :]
                    loss = criterion(output.view(-1, output.shape[-1]), tgt_out.view(-1))

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), args.clip_norm)

                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            # if accelerator.is_main_process: 
            #     progress_bar.set_postfix({"Loss": loss.item(), "LR": scheduler.get_last_lr()})
        
        if accelerator.is_main_process:
            print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader)}")

        # Save model
        if epoch % args.save_interval == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                
                checkpoint_fname = os.path.join(args.save_dir, f"epoch_{epoch+1}_bs_{args.batch_size}.pt")
                accelerator.save(unwrapped_model.state_dict(), checkpoint_fname)

if __name__ == '__main__':
    args = train_arg()
    main(args)
