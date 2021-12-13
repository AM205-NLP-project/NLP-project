import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    json_dat = json.load(open('wn_dat/ex5.json'))

    # creates the labels of the various meanings
    labels = [f'{x["words"].replace(";", "; ")}' for x in json_dat]
    le = LabelEncoder()
    label_ids = le.fit_transform(labels)

    # path = 'wn_dat/ex5_False_12_avg.pkl'
    # path = 'wn_dat/ex5_False_12_last.pkl'
    # path = 'wn_dat/ex5_True_12_last.pkl'
    path = 'wn_dat/ex5_True_12_avg.pkl'

    dat = torch.load(path)  # pre-computed vectors
    dat = dat.detach()  # removes gradient requirement
    dat = dat.numpy()  # transforms to simple numpy matrix

    # number of classes to sub-sample for visualization and dimensionality reduction
    # sampling a subset or else its too much to visualize...
    classes = 18
    subsel = set(np.random.choice(list(range(len(le.classes_))), size=classes, replace=False))
    keep = np.full(dat.shape[0], False)
    for ix, idx in enumerate(label_ids):
        if idx in subsel:
            keep[ix] = True
    dat = dat[keep, :]
    label_ids = label_ids[keep]

    # scale the features to a standard normal
    scaler = StandardScaler()
    scaled_dat = scaler.fit_transform(dat)

    # reduce dimensionality
    reducer = TSNE()
    red_dat = reducer.fit_transform(scaled_dat)

    # plot
    cs = plt.cm.get_cmap('tab20', len(le.classes_))
    plt.figure(figsize=(8, 6))
    for ix in range(len(le.classes_)):
        if ix not in subsel:
            continue

        xs = red_dat[label_ids == ix, :]
        plt.scatter(xs[:, 0], xs[:, 1], label=le.inverse_transform([ix])[0], c=cs(ix))

    lgd = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # plt.tight_layout()
    # plt.show()

    plt.savefig('TSNE', bbox_extra_artists=(lgd,), bbox_inches='tight')

    # import pdb
    # pdb.set_trace()
