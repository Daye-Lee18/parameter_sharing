#!/usr/bin/env python3

from arg_parse import train_arg as train_arg
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

def main(args):
    # data preprocessing 
    # BPE model training for data 
    data_dir = args.data
    model_file=os.path.join(data_dir, "bpe.model")
    
    if not os.path.exists(model_file):
        print(f"Training A SentencePiece model")
        combined_train_file = os.path.join(data_dir, "train.en-de.en.bin")
        
        spm.SentencePieceTrainer.train(
            input=combined_train_file,
            model_prefix=os.path.join(data_dir, "bpe"),
            vocab_size=32000,
            character_coverage=1.0,
            model_type="bpe"
        )
    # Load the SentencePiece model
    src_sp = spm.SentencePieceProcessor(model_file=os.path.join(data_dir, "bpe.model"))
    tgt_sp = spm.SentencePieceProcessor(model_file=os.path.join(data_dir, "bpe.model"))

    # build a dataset and dataloader 
    train_dataset = TranslationDataset(os.path.join(data_dir, 'train.en-de.en.bin'), os.path.join(data_dir, 'train.en-de.de.bin'), src_sp, tgt_sp)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: x)
 
    # INIT 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_token = src_sp.pad_id()
    model = ParameterShareTransformer(d_model=512, nhead=16, num_encoder_layers=6, num_decoder_layers=6,
                                      dim_feedforward=2048, dropout=0.1, activation="relu", num_unique_encoder_layers=3,
                                      num_unique_decoder_layers=3, mode="cycle_rev")


    # criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=eval(args.adam_betas), weight_decay=args.weight_decay)
    criterion = LabelSmoothingLoss(smoothing=args.label_smoothing).to(device)
    scheduler = InverseSqrtScheduler(optimizer, args.warmup_updates, args.warmup_init_lr, args.lr)
    scaler = GradScaler()
    print(f"Total number of parameters of the model: {sum(p.numel() for p in model.parameters())}")

    # Training 
    model.train()
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            src_batch, tgt_batch = zip(*batch)
            src_batch = pad_sequence(src_batch, padding_value=pad_token).to(device)
            tgt_batch = pad_sequence(tgt_batch, padding_value=pad_token).to(device)

            tgt_input = tgt_batch[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_batch, tgt_input, pad_token)

            optimizer.zero_grad()
            with autocast():
                output = model(src_batch, tgt_input, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, src_padding_mask)
                tgt_out = tgt_batch[1:, :]
                loss = criterion(output.view(-1, output.shape[-1]), tgt_out.view(-1))
            
            # loss.backward()
            scaler.scale(loss).backward()
            
            if args.clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            
            # progress_bar.set_postfix({"Loss": loss.item(), "LR": optimizer.param_groups[0]["lr"]})
            progress_bar.set_postfix({"Loss": loss.item(), "LR": scheduler.step()})

        print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader)}")

if __name__ == '__main__':
    args = train_arg()
    main(args)