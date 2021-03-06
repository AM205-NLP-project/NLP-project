import ast
import json

import numpy as np
import pandas as pd

from tqdm import tqdm


def preprocess(
        data,
        name='toy',
        train_split=0.7, val_split=0.2, random_state=1234,
        special_syms=('Word: ', 'Def: ', 'Ex: '),
):
    # set random seed
    np.random.seed(random_state)

    # shuffle the data
    np.random.shuffle(data)

    # make splits
    train_cnt = int(train_split * len(data))
    val_cnt = int(val_split * len(data))
    splits = {
        'train': data[:train_cnt],
        'val': data[train_cnt:train_cnt+val_cnt],
        'test': data[train_cnt+val_cnt:]
    }

    # create files
    idx = [0, 1, 2]
    for k, vs in splits.items():
        with open(f'./{name}_{k}.txt', 'w+') as fp:
            for v in vs:
                # np.random.shuffle(idx)
                line = ''
                for ix in idx:
                    line += special_syms[ix] + v[ix] + '<|endoftext|>'
                line += f'\n'
                fp.write(line)


def preprocess_pd(
        path,
        train_split=0.8, val_split=0.1, random_state=1234,
        word_sym='Word: ', def_sym='Definition: ', ex_sym='Example: ',
        pad_sym=' ', eos_sym='<|endoftext|>',
        rev=False,
):
    # set random seed
    np.random.seed(random_state)

    # read data
    df = pd.read_csv(path)

    samples = []
    ex_cache = set()
    def_cache = set()
    # flatten into list of (def, word) & (ex, word) pairs
    for ix, row in tqdm(df.iterrows(), desc='Parsing DataFrame into samples...', total=len(df)):
        # unpack features
        word, pos, defs, rels, exs = row

        # parse lists as strings into actual lists
        defs = ast.literal_eval(defs)
        exs = ast.literal_eval(exs)

        # add all definitions
        for d in defs:
            # if rev:
            #     samples.append(f'{word_sym}{word}{pad_sym}{def_sym}{d}{eos_sym}')
            # else:
            #     samples.append(f'{def_sym}{d}{pad_sym}{word_sym}{word}{eos_sym}')
            if not d:
                continue

            if word == d:
                continue

            if d in def_cache:
                continue

            if len(d) < 4:
                print('Def: ', d)
                import pdb
                pdb.set_trace()
            samples.append({
                'word': word_sym + word,
                'text': def_sym + d,
            })

            def_cache.add(d)

        # add all examples
        for e in exs:
            # if rev:
            #     samples.append(f'{word_sym}{word}{pad_sym}{ex_sym}{e}{eos_sym}')
            # else:
            #     samples.append(f'{ex_sym}{e}{pad_sym}{word_sym}{word}{eos_sym}')

            if e in ex_cache:
                continue

            elen = len(e)
            if word in e and len(word) / elen > 0.5:
                continue

            if elen < 6:
                if 'Audio' == e:
                    continue
                continue
            elif elen > 512:
                continue

            samples.append({
                'word': word_sym + word,
                'text': ex_sym + e,
            })

            ex_cache.add(e)

    print(f'{len(samples)} samples found.')

    # shuffle samples
    np.random.shuffle(samples)

    # make splits
    train_cnt = int(train_split * len(samples))
    val_cnt = int(val_split * len(samples))
    splits = {
        'train': samples[:train_cnt],
        'val': samples[train_cnt:train_cnt + val_cnt],
        'test': samples[train_cnt + val_cnt:]
    }

    for k, vs in splits.items():
        print(len(vs), 'samples in', k)
        json.dump(vs, open(f'./{path}_{k}.json', 'w+'), indent=2)
        # with open(f'./{path}{"_rev" if rev else ""}_{k}.txt', 'w+') as fp:
        #     for v in tqdm(vs, desc=k, total=len(vs)):
        #         fp.write(v + '\n')


if __name__ == '__main__':
    # toy_example = [
    #     (
    #         'learn',
    #         'to acquire, or attempt to acquire knowledge or an ability to do something.',
    #         'every day I learn more about this great city.'
    #     ),
    #     (
    #         'learn',
    #         'to attend a course or other educational activity.',
    #         'for, as he took delight to introduce me, I took delight to learn.'
    #     ),
    #     (
    #         'learn',
    #         'to gain knowledge from a bad experience so as to improve.',
    #         'learn from one\'s mistakes'
    #     ),
    #     (
    #         'learn',
    #         'to study',
    #         'i learn medicine.',
    #     ),
    #     (
    #         'learn',
    #         'to study',
    #         'they learn psychology.',
    #     ),
    #     (
    #         'learn',
    #         'to come to know; to become informed of; to find out.',
    #         'he just learned that he will be sacked.'
    #     ),
    #     (
    #         'pizza',
    #         '(uncountable) a baked Italian dish of a thinly rolled bread dough crust typically topped before baking with tomato sauce, cheese and other ingredients such as meat, vegetables or fruit',
    #         'a slice of pizza'
    #     ),
    #     (
    #         'pizza',
    #         '(uncountable) a baked Italian dish of a thinly rolled bread dough crust typically topped before baking with tomato sauce, cheese and other ingredients such as meat, vegetables or fruit',
    #         'a pizza pie'
    #     ),
    #     (
    #         'pizza',
    #         '(uncountable) a baked Italian dish of a thinly rolled bread dough crust typically topped before baking with tomato sauce, cheese and other ingredients such as meat, vegetables or fruit',
    #         'want to go out for pizza tonight?'
    #     ),
    #     (
    #         'pizza',
    #         '(countable) a single instance of this dish',
    #         'he ate a whole pizza!'
    #     )
    # ]
    # preprocess(toy_example)

    preprocess_pd('words_dataset_unique.csv')
    # preprocess_pd('words_dataset', rev=True)
