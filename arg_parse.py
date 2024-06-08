import argparse 

def train_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/data-bin/wmt16_en_de_bpe32k", help="WMT'16 data directory")
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--adam-betas", default='(0.9, 0.98)')
    parser.add_argument("--clip-norm", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr-scheduler", type=str, default="inverse_sqrt")
    parser.add_argument("--warmup-updates", type=int, default=4000)
    parser.add_argument("--warmup-init-lr", type=float, default=1e-07)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--criterion", type=str, default="label_smoothed_cross_entropy")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=3584)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    return args 
