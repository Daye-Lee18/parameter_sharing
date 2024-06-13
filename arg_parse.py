import argparse 

def train_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="cycle_rev")
    parser.add_argument("--input_dim", type=int, default=32000)
    parser.add_argument("--output_dim", type=int, default=32000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--src_pad_idx", type=int, default=3)
    parser.add_argument("--tgt_pad_idx", type=int, default=3)
    parser.add_argument("--src_lang", type=str, default="en", help="source language")
    parser.add_argument("--tgt_lang", type=str, default="de", help="target language")
    parser.add_argument("--data", type=str, default="data/data-bin/wmt16_en_de_bpe32k", help="WMT'16 data directory")
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--adam-betas", default='(0.9, 0.98)')
    parser.add_argument("--clip-norm", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr-scheduler", type=str, default="inverse_sqrt")
    parser.add_argument("--warmup-updates", type=int, default=4000)
    parser.add_argument("--warmup-init-lr", type=float, default=1e-07)
    
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--dim_feedforward", type=int, default=1024)

    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--criterion", type=str, default="label_smoothed_cross_entropy")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=3584)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--encoder-layer", type=int, default=8)
    parser.add_argument("--decoder-layer", type=int, default=8)

    # resuming the training 
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--wandb_pj_name", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="")


    args = parser.parse_args()

    return args 
