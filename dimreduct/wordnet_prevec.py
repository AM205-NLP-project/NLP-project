import json

import torch

from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel

from tqdm import tqdm


raw = json.load(open('wn_dat/ex5.json'))

tok = AutoTokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('../tune/model_unique_best/')

xs = torch.zeros((len(raw), 768))
for ex in tqdm(raw):
    model_inp = tok(ex['example'], return_tensors='pt')
    output = model(**model_inp, output_hidden_states=True)

    final_rep = output.hidden_states[-1]
    avg_vec = final_rep.squeeze().mean(axis=0)
    xs[ex['sample_id'], :] = avg_vec

torch.save(xs, 'wn_dat/ex5_avg.pkl')