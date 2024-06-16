# parameter_sharing
This is a repository for replicating &lt;Lessons on Parameter Sharing across Layers in Transformers> proposed in 2021 

# Accelerate config 
```bash
accelerate config --config_file accel.yaml
```
- make sure to choose fp16 setting 

# Data Download 

1. Download The WMT 2016 English-German dataset, Train the BPE tokenizer, and Tokenize the dataset 
```bash
python ./data/data_downloader.py
```

After running the above code, your directory should look like the structure below:

parameter_sharing
├── data
│   └── data-bin
│       └── wmt16_en_de_bpe32k
└── model


# Training 

```bash
accelerate launch --config_file accel.yaml --main_process_port 27013 train.py \
    --batch_size 32 \
    --data data/data-bin/wmt16_en_de_bpe32k \
    --input_dim 32000 \
    --output_dim 32000 \
    --epochs 1000 \
    --src_pad_idx 3 \
    --tgt_pad_idx 3 \
    --src_lang en \
    --tgt_lang de \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 1 \
    --lr 0.0005 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 512 \
    --save_interval 200 \
    --save_dir /home/s2/dayelee/dayelee_store/paramShareTrans/checkpoints \

```

