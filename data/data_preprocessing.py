import os
import sentencepiece as spm

# data path directory setting 
data_dir = "data/wmt16_en_de"
train_en = os.path.join(data_dir, "train.en")
train_de = os.path.join(data_dir, "train.de")

# combining the training data (merging the source and target data for BPE training)
combined_train_file = os.path.join(data_dir, "train.en-de")
with open(combined_train_file, "w", encoding="utf-8") as outfile:
    for file in [train_en, train_de]:
        with open(file, "r", encoding="utf-8") as infile:
            for line in infile:
                outfile.write(line)

# Traing the SentencePiece model
spm.SentencePieceTrainer.train(
    input=combined_train_file,
    model_prefix=os.path.join(data_dir, "bpe"),
    vocab_size=32000,
    character_coverage=1.0,
    model_type="bpe"
)

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor(model_file=os.path.join(data_dir, "bpe.model"))

# tokenize text files by BPE 
def tokenize_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            tokens = sp.encode(line.strip(), out_type=str)
            outfile.write(" ".join(tokens) + "\n")

# tokenize the training data 
tokenize_file(train_en, os.path.join(data_dir, "train.bpe.en"))
tokenize_file(train_de, os.path.join(data_dir, "train.bpe.de"))

# # 검증 및 테스트 데이터 파일 경로 설정 (필요한 경우)
# valid_en = os.path.join(data_dir, "valid.en")
# valid_de = os.path.join(data_dir, "valid.de")
# test_en = os.path.join(data_dir, "test.en")
# test_de = os.path.join(data_dir, "test.de")

# # 검증 및 테스트 데이터 토큰화 (필요한 경우)
# tokenize_file(valid_en, os.path.join(data_dir, "valid.bpe.en"))
# tokenize_file(valid_de, os.path.join(data_dir, "valid.bpe.de"))
# tokenize_file(test_en, os.path.join(data_dir, "test.bpe.en"))
# tokenize_file(test_de, os.path.join(data_dir, "test.bpe.de"))
