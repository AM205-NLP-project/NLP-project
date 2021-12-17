# NLP-project

This project contains the code 
that was used to finetune a GPT-2 model 
on Wiktionary data. 
Specifically, 
two models were generated:
- Forward: A model that can take a word and generate an example usage or a definition
  - Link to download the weights: [https://drive.google.com/file/d/1KBNXQX8Ra9wOH79o8Gy4XS0hhHWrS-zo/view?usp=sharing](https://drive.google.com/file/d/1KBNXQX8Ra9wOH79o8Gy4XS0hhHWrS-zo/view?usp=sharing)
- Reverse: A model that can take a definition and output potential words
  - Link to download the weights: [https://drive.google.com/file/d/12uH8wx-dwWInQszWw7a7ivV5KVAi8CmQ/view?usp=sharing](https://drive.google.com/file/d/12uH8wx-dwWInQszWw7a7ivV5KVAi8CmQ/view?usp=sharing)

`Demo.ipynb` contains a demo for using the models. 

## Tuning 

The code for finetuning GPT-2 can be found in `tune/`.

Some of the files therein: 
- `benchmark.py` -- Code for benchmarking the models on the train/validation/test splits of the dataset
- `demoing.py` -- Simple demo script for sampling and extracting features from the model
- `get_vec.py` -- Simple script for extracting a "word vector" from our model
- `guess.py` -- Script for running the guessing game on defnitions 
- `main.py` -- Script for actually fine-tuning GPT-2 on the Wiktionary data 
- `preprocess.py` -- Script that takes the raw data obtained from Wiktionary and generates the splits of the dataset 
- `sampler.py` -- A script that is specifically for sampling from our tuned model 