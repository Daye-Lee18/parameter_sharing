# parameter_sharing
This is a repository for replicating &lt;Lessons on Parameter Sharing across Layers in Transformers> proposed in 2021 


# Data Download 

1. Download The WMT 2016 English-German training dataset 
```bash
cd ./data
# change OUTPUT_DIR in download.sh file as follows 
# OUTPUT_DIR=/path/to/your/output_dir
bash download.sh
```

2. Preprocess the dataset with a joined dictionary 

```bash

OUTPUT_DIR=/path/to/your/output_dir

fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $OUTPUT_DIR/train.tok.clean.bpe.32000 \
    --validpref $OUTPUT_DIR/newstest2013.tok.bpe.32000 \
    --testpref $OUTPUT_DIR/newstest2014.tok.bpe.32000 \
    --destdir data-bin/wmt16_en_de_bpe32k \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20

```

After running the above code, your directory should look like the structure below:

parameter_sharing
├── data
│   └── data-bin
│       └── wmt16_en_de_bpe32k
└── model


# Training 

```bash
train.py --data data-bin/wmt16_en_de_bpe32k \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 0.0005 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 \
    --fp16
```

