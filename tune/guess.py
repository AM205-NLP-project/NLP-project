import json

import torch

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel


def guess_words(x, guesses, maxlen=8, verbose=False):
    """
    Simple function for guessing a word based on beam search

    :param x: The text of the example / definition
    :param guesses: The number of guesses
    :param maxlen: The maximum amount of tokens to generate per guess
    :param verbose: Whether or not to print the examples
    :return: (guesses, metrics)
    """
    seed_txt = x['text'] + ' ; Word:'

    input_ids = tok.encode(seed_txt, return_tensors='pt')
    input_ids = input_ids.to(device)

    beam_outputs = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + maxlen,
        num_beams=guesses,
        num_return_sequences=guesses,
        early_stopping=True,
        pad_token_id=tok.eos_token_id,
    )

    words = []
    true_word = x['word'].replace('Word: ', '')
    if verbose:
        print('Seed:', seed_txt)
        print(f"Guesses (True Answer: {true_word}):\n" + 100 * '-')

    for i, beam_output in enumerate(beam_outputs):
        w = tok.decode(beam_output, skip_special_tokens=True).replace(seed_txt, '').strip()
        words.append(w)

        if verbose:
            print("{}: {}".format(i, w))

    # compute various versions of mean reciprocal rank
    metrics = {
        'Exact MRR': max([1 / (ix + 1) if w == true_word else 0 for ix, w in enumerate(words)]),
        'Recall MRR': max(
            [(1 / (ix + 1)) * min(len(w) / len(true_word), 1) if w in true_word else 0 for ix, w in enumerate(words)]),
        'Precision MRR': max(
            [(1 / (ix + 1)) * min(len(true_word) / len(w), 1) if true_word in w else 0 for ix, w in enumerate(words)]),
    }

    return words, metrics


splits = [
    'train',
    'val',
    'test'
]
cnts = {k: 0 for k in splits}
data = {
    k: json.load(open(f'words_dataset_unique.csv_{k}.json'))
    for k in splits
}

model = GPT2LMHeadModel.from_pretrained('model_unique_rev_best/')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

tok = AutoTokenizer.from_pretrained('gpt2')
tok.add_special_tokens({'pad_token': '<|endoftext|>'})

for split in data:
    print(split)
    kx = 10  # number of guesses per definition

    # allows re-loading if evaluation
    # only partially performed
    try:
        with open(f'gg_res_{split}.json') as fp:
            for lx in fp.readlines():
                cnts[split] += 1
    except FileNotFoundError:
        pass

    for ix, dx in enumerate(tqdm(data[split])):
        if ix < cnts[split]:
            continue

        ws, ms = guess_words(dx, kx)
        with open(f'gg_res_{split}.json', 'a+') as fp:
            fp.write(json.dumps(ms) + '\n')
