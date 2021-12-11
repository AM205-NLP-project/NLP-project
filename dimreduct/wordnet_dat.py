import json

from collections import defaultdict
from nltk.corpus import wordnet as wn

ex_thresh = 5  # example cut-off threshold; if a synset doesn't have this minimally, we toss

dat = defaultdict(dict)
lem2syn = defaultdict(set)
for synset in wn.all_synsets('v'):

    # skip synset if does not have min # of examples
    if len(synset.examples()) < ex_thresh:
        continue

    # build structure
    dat[synset.name()]['words'] = [lem.name().split('.')[-1].replace('_', ' ') for lem in synset.lemmas()]
    dat[synset.name()]['definition'] = synset.definition()
    dat[synset.name()]['examples'] = synset.examples()

    # record word -> synset map
    for lem in dat[synset.name()]['words']:
        lem2syn[lem].add(synset.name())

# do a check to retain only synsets with
# at least 1 polysemous verb
# i.e., the verb must appear in at least 2 synsets
poly_lem = 0
poly_syns = set()
for lem, syns in lem2syn.items():
    if len(syns) > 1:
        poly_lem += 1
        for s in syns:
            poly_syns.add(s)

# count the # examples included in the polysemous subset
exs = 0
for syn in poly_syns:
    exs += len(dat[syn]['examples'])

# print # of poly synsets, # of examples
print(len(poly_syns), exs)

# construct dataset
dataset = []
ix = 0
for syn in dat:
    if syn in poly_syns:  # only if synset is in the polysemous subset
        word_list = ';'.join(dat[syn]['words'])
        for ex in dat[syn]['examples']:
            dataset.append({
                'sample_id': ix,
                'synset_id': syn,
                'words': word_list,
                'definition': dat[syn]['definition'],
                'example': ex
            })

            ix += 1

json.dump(dataset, open(f'wn_dat/ex{ex_thresh}.json', 'w+'), indent=2)

import pdb
pdb.set_trace()
