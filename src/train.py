import pandas as pd
from DeepPurpose import CompoundPred

from .models import get_model


def train(model_name, train_csv, val_csv, model_dir, **model_kwargs):
    model = get_model(model_name, **model_kwargs)

    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)

    model.train(train, val)

    model.save_model(model_dir)

    return model
