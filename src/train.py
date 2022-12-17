import pandas as pd
from DeepPurpose import CompoundPred

from .models import get_model


def train(model_name, train_pickle, val_pickle, model_dir, **model_kwargs):
    model = get_model(model_name, **model_kwargs)

    train_df = pd.read_pickle(train_pickle)
    val_df = pd.read_pickle(val_pickle)

    model.train(train_df, val_df)

    model.save_model(model_dir)

    return model
