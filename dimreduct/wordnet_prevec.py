import json

import torch

from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel

from tqdm import tqdm

ex_thresh = 5
reverse = True  # whether to use the reverse model (Example/Definition -> Word)
layer = 12  # which layer to use \in {0, 1, 2, ..., 12}

mode = 'avg'  # take the average across the latent dimension
# mode = 'last'  # take the last vector

raw = json.load(open(f'wn_dat/ex{ex_thresh}.json'))

tok = AutoTokenizer.from_pretrained('gpt2')
if reverse:
    model = GPT2LMHeadModel.from_pretrained('../tune/model_unique_rev_best/')
else:
    model = GPT2LMHeadModel.from_pretrained('../tune/model_unique_best/')

xs = torch.zeros((len(raw), 768))
for ex in tqdm(raw):

    # formatting the text sequence
    text_inp = 'Example: ' + ex['example'] + ' ; Word:'
    # text_inp = 'Definition: ' + ex['definition'] + ' ; '

    model_inp = tok(text_inp, return_tensors='pt')
    output = model(**model_inp, output_hidden_states=True)

    final_rep = output.hidden_states[layer]
    if mode == 'avg':
        vec = final_rep.squeeze().mean(axis=0)
    elif mode == 'last':
        vec = final_rep.squeeze()[-1]
    else:
        raise ValueError

    xs[ex['sample_id'], :] = vec

torch.save(xs, f'wn_dat/ex{ex_thresh}_{reverse}_{layer}_{mode}.pkl')
