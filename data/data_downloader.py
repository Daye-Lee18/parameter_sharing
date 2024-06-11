from datasets import load_dataset # huggingface 
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors 
import os 

# reference 
# https://huggingface.co/docs/tokenizers/en/quicktour
# https://www.kaggle.com/code/vimalpillai/training-bpe-tokenizer

# Load datasets for training the tokenizer
def get_training_corpus(dataset):
    for i in range(0, len(dataset['train']), 1000):
        # import pdb; pdb.set_trace()
        # Dataset['train'][i:i+1000]이 dictionary라 for loop안에 못들어갔음 
        # yield [item['translation']['en'] for item in dataset['train'][i: i + 1000]] + [item['translation']['de'] for item in dataset['train'][i: i + 1000]]
        yield [item['en'] for item in dataset['train'][i: i + 1000]['translation']] + [item['de'] for item in dataset['train'][i: i + 1000]['translation']]

# Function to encode using the tokenizer
def encode(examples):

    tokenizer = Tokenizer.from_file("data/data-bin/wmt16_en_de_bpe32k/tokenizer.json")
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )# Setup post-processor
    
    src = [item['en'] for item in examples['translation']]
    tgt = [item['de'] for item in examples['translation']]
    
    src_encodings = tokenizer.encode_batch(src)
    tgt_encodings = tokenizer.encode_batch(tgt)
    
    src_encoded = [" ".join(map(str, enc.ids)) for enc in src_encodings]
    tgt_encoded = [" ".join(map(str, enc.ids)) for enc in tgt_encodings]
    
    return {'src_encoded': src_encoded, 'tgt_encoded': tgt_encoded}

# Function to save encoded data
def save_encoded_data(encoded_data, src_filename, tgt_filename):
    with open(src_filename, 'w') as src_f, open(tgt_filename, 'w') as tgt_f:
        for src_item, tgt_item in zip(encoded_data['src_encoded'], encoded_data['tgt_encoded']):
            src_f.write(src_item + "\n")
            tgt_f.write(tgt_item + "\n")

if __name__ == "__main__":
    # Load the WMT16 dataset 
    dataset = load_dataset("wmt16", "de-en") # train, validation, test 

    # Initialize a tokenizer 
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() # Setup pre-tokenizer
    special_tokens=["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"]
    PAD_IDX = special_tokens.index("[PAD]")
    print(f"PAD_IDX: {PAD_IDX}")


    #########  Training on Dataset 
    # Setup a trainer for BPE
    trainer = trainers.BpeTrainer(vocab_size=32000, min_frequency = 2, special_tokens=special_tokens)
    # Train the tokenizer

    tokenizer.train_from_iterator(get_training_corpus(dataset), trainer=trainer)

    # Set decoder
    tokenizer.decoder = decoders.BPEDecoder()

    # Create directories to save the tokenized data
    os.makedirs("data/data-bin/wmt16_en_de_bpe32k", exist_ok=True)

    # Save the tokenizer
    tokenizer.save("data/data-bin/wmt16_en_de_bpe32k/tokenizer.json")

    ######### encode the train, validation, test Dataset 
    # Apply the tokenizer to the dataset
    encoded_dataset = dataset.map(encode, batched=True)

    # Save encoded training, validation, and test sets
    save_encoded_data(encoded_dataset['train'], "data/data-bin/wmt16_en_de_bpe32k/train.en.bpe", "data/data-bin/wmt16_en_de_bpe32k/train.de.bpe")
    save_encoded_data(encoded_dataset['validation'], "data/data-bin/wmt16_en_de_bpe32k/valid.en.bpe", "data/data-bin/wmt16_en_de_bpe32k/valid.de.bpe")
    save_encoded_data(encoded_dataset['test'], "data/data-bin/wmt16_en_de_bpe32k/test.en.bpe", "data/data-bin/wmt16_en_de_bpe32k/test.de.bpe")