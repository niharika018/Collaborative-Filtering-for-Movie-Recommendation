import numpy as np


def hitratio(ratings, top_n=10):
    hr = 0.0
    for r in ratings:
        pos_index = len(r) - 1
        # get top n indices
        arg_index = np.argsort(-r)[:top_n]
        if pos_index in arg_index:
            # increment
            hr += 1
    return hr / len(ratings)


def ndcg(ratings, top_n=10):
    ndcg = 0.0
    for r in ratings:
        pos_index = len(r) - 1
        # get top n indices
        arg_index = np.argsort(-r)[:top_n]
        if pos_index in arg_index:
            # get the position
            ndcg += np.log(2.0) / np.log(arg_index.tolist().index(pos_index) + 2.0)
    return ndcg / len(ratings)
