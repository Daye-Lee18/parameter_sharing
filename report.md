Daye Lee 

2-3 pages long 
# Report 

## Introduction 
  - I replicated one of the parameter-sharing methods proposed in \**<Lessons on Parameter Sharing across Layers in Transformers>**
  - The task was to translate English into German which were measured on WMT 2014 English-Genman dataset
  - Proposed new three parameter sharing methods (SEQUENCE, CYCLE, CYCLE (REV)) for Transformers which are efficient in terms of the model size and computational time 
  - **Current State-of-Art** Model on BLEU metric for WMT 2014 English-German Dataset 
    - The model introduced in this paper is the current state-of-art model on BLUE metric for WMT 2014 English-German dataset
  - **Parameter sharing for Smaller Model Size**
    - To reduce computational training time, it is essential to decrease the size of models without compromising their performance. Parameter sharing is an effective method to achieve this. 
  - **Less Computation Cost**
    - The proposed parameter sharing architecture in this paper enables the creation of smaller models without sacrificing, thereby reducing training time. This is crucial in the era of large-scale models  

## Methodology 
  - Detail the approach taken in the original paper and the steps you followed to replicate it 
  - Proposed parameter sharing methods: prepare parameters for M layers to construct N layered Transformer-based encoder-decoder when M = 3 and N = 6 (Where 1 <= M <= N)
  - Out of three parameter-sharing methods, I replicated the CYCLE method 
  - Algorithm: 
    - **SEQUENCE**: two sequential layers share their parameters as illustrated in Figure 1 
    - **CYCLE**: 
      - stack M layers whose parameters are independent from each other 
      - repeat 
        - stacking the M layers with the identical order to the first M layers until the total number of layers reaches N 
    - **CYCLE (REV)**: 
      - repeat stacking the M layers as CYCLE until M * (N/M -1) layers
      - for the remaining layers, stack M layers in the reverse order
  - Include any modifications you made and the reasons for them 

## Experiments on Machine Translation 
  - Training dataset: WMT 2016 training dataset (4.5 M English-German sentence pairs) 
  - Data download and data pre-processing (BPE):
    - `data/data_downloader.py`
    - build a vocabulary set with BPE (Byte Pair Encoding) (Sennrich et al., 2016b)
    - set the number of BPE merge operations at 32K and shared the vocabulary between the source and target languages.
  - Dataset load and dataloader 
    - `data/dataloader.py`
  - Model Training 
    - encoder & decoder with M sharing layers, made the total of N layers, which is implemented in the `model/ParameterShareTransformer.py`
    - `train.py`
    - experiments setting 
      - GPU 4, epoch 1000, 
  - Hyperparameter setting 
    - epoch, ,,,,
    
## Results 
  - Metric:
    - BLEU:
    - sacreBLEU: case-sensitive detokenized BLEU (Post, 2018)
  - Present your results, comparing them to those reported in the original paper 
  - Include tables, charts, or other visual aids as necessary 

## Discussion 
  - Analyze the results, discussing any discrepencies between your results and those of the original paper 
  - Highlight any challengs you encounted and how you addressed them 

## Conclusion 
  - Summarize your findings and reflect on the replication process 
  - Discuss the implications of your work and any potential future directions 


-------------

## Note 
1. Universal Transformers: 
  - Dehghani et al. 2019 
  - "All" layers sharing: consists of parameters for only one layer of a Transformer-based encoder-decoder, and uses these parameters N times for an N-layered encoder-decoder 
  - cons: big feature dimension, computational inefficient than normal Transformer 


## Reference 
Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016b. Neural machine translation of rare words with subword units. In Proceedings of ACL, pages 1715–1725.

Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of WMT, pages 186–191.