import numpy as np


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


if __name__ == '__main__':
    toy_example = [
        (
            'learn',
            'to acquire, or attempt to acquire knowledge or an ability to do something.',
            'every day I learn more about this great city.'
        ),
        (
            'learn',
            'to attend a course or other educational activity.',
            'for, as he took delight to introduce me, I took delight to learn.'
        ),
        (
            'learn',
            'to gain knowledge from a bad experience so as to improve.',
            'learn from one\'s mistakes'
        ),
        (
            'learn',
            'to study',
            'i learn medicine.',
        ),
        (
            'learn',
            'to study',
            'they learn psychology.',
        ),
        (
            'learn',
            'to come to know; to become informed of; to find out.',
            'he just learned that he will be sacked.'
        ),
        (
            'pizza',
            '(uncountable) a baked Italian dish of a thinly rolled bread dough crust typically topped before baking with tomato sauce, cheese and other ingredients such as meat, vegetables or fruit',
            'a slice of pizza'
        ),
        (
            'pizza',
            '(uncountable) a baked Italian dish of a thinly rolled bread dough crust typically topped before baking with tomato sauce, cheese and other ingredients such as meat, vegetables or fruit',
            'a pizza pie'
        ),
        (
            'pizza',
            '(uncountable) a baked Italian dish of a thinly rolled bread dough crust typically topped before baking with tomato sauce, cheese and other ingredients such as meat, vegetables or fruit',
            'want to go out for pizza tonight?'
        ),
        (
            'pizza',
            '(countable) a single instance of this dish',
            'he ate a whole pizza!'
        )
    ]
    preprocess(toy_example)
