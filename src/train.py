import pandas as pd

from .models import get_model


def train(model_name, train_pickle, val_pickle, model_dir, **model_kwargs):
    """
    Train a model.

    :param model_name: either 'Transformer', 'DGL_GCN' (Graph Convolution Network)
        or 'tree' (decistion tree model).
    :param train_pickle: path to a pickle with the train dataset.
    :param val_pickle: path to a pickle with the validation dataset.
    :param model_dir: path to a directory to save trained model to.

    :return: model.
    """
    model = get_model(model_name, **model_kwargs)

    train_df = pd.read_pickle(train_pickle)
    val_df = pd.read_pickle(val_pickle)

    model.train(train_df, val_df)

    model.save_model(model_dir)

    return model
