OUTPUT_DIR=/home/s2/dayelee/dayelee_store/WMT16

fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $OUTPUT_DIR/train.tok.clean.bpe.32000 \
    --validpref $OUTPUT_DIR/newstest2013.tok.bpe.32000 \
    --testpref $OUTPUT_DIR/newstest2014.tok.bpe.32000 \
    --destdir data-bin/wmt16_en_de_bpe32k \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20
