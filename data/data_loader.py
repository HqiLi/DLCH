import h5py
import scipy.io as scio
import numpy as np

np.set_printoptions(threshold=np.inf)


def load_data(path):
    file = h5py.File(path, 'r')
    images = file['images'][:].astype('float32')
    labels = file['LAll'][:]
    tags = file['YAll'][:]

    file.close()
    return images, tags, labels


def load_pretrain_model(path):
    return scio.loadmat(path)


def split_data(images, tags, labels, num_query, num_seen, seed=None):
    np.random.seed(seed)
    random_index = np.random.permutation(range(20015))
    query_index = random_index[: num_query]
    retrieval_index = random_index[num_query:]

    # split seen, unseen
    unseen_index = np.array([])
    seen_index = np.array([])
    for i in retrieval_index:
        t = labels[i]
        jug = 0
        for j in range(num_seen, 24):
            if t[j] == 1:
                jug = jug + 1
        if jug == 0:
            seen_index = np.append(seen_index, i)
        else:
            unseen_index = np.append(unseen_index, i)

    unseen_index = unseen_index.astype('int64')
    seen_index = seen_index.astype('int64')

    X = {}
    X['query'] = images[query_index]
    X['unseen'] = images[unseen_index]
    X['seen'] = images[seen_index]
    X['retrieval'] = images[retrieval_index]

    Y = {}
    Y['query'] = tags[query_index]
    Y['seen'] = tags[seen_index]
    Y['unseen'] = tags[unseen_index]
    Y['retrieval'] = tags[retrieval_index]

    L = {}
    L['query'] = labels[query_index]
    L['seen'] = labels[seen_index]
    L['unseen'] = labels[unseen_index]
    L['retrieval'] = labels[retrieval_index]

    return X, Y, L


def sample_data(images, tags, labels, num_seendata, num_samples):
    # sample training data
    unseen_in_unseen_index = np.array([])
    unseen_in_sample_index = np.array([])
    num_retrieval = len(images)

    train_index = np.random.permutation(num_retrieval)[:num_samples]
    unseen_sample_in_unseen_index = train_index[train_index > num_seendata] - num_seendata
    unseen_sample_in_sample_index = (train_index > num_seendata).nonzero()[0]

    train_x = images[train_index]
    train_y = tags[train_index]
    train_L = labels[train_index]

    return train_L, train_x, train_y, train_index, unseen_in_unseen_index, unseen_in_sample_index