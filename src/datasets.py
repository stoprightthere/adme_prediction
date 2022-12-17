import pandas as pd

from DeepPurpose import utils
from tdc.single_pred import ADME


_DEEPPURPOSE_MODELS = ['Transformer', 'DGL_GCN']


def prepare_dataset(dataset_name, model_name):
    """
    Prepare a dataset. That is, add necessary drug encoding based on `model_name`, 
        split the dataset into train, validation, and test sub-datasets.

    :param dataset_name: name of the dataset on the TDC.
    :param model_name:  either 'Transformer', 'DGL_GCN' (Graph Convolution Network) or 'tree' (decistion tree model).

    :return: (train, val, test) datasets.
    """
    data = ADME(name=dataset_name)
    X, y = data.get_data(format = 'DeepPurpose')

    if model_name in _DEEPPURPOSE_MODELS:
        train, val, test = utils.data_process(
            X_drug = X, 
            y = y, 
            drug_encoding = model_name,
            random_seed = 'TDC',
        )
    elif model_name == 'tree':
        df = pd.DataFrame(zip(X, y))
        df.rename(columns={0: 'SMILES', 1: 'Label'}, inplace=True)
        df = utils.encode_drug(df, 'Pubchem')
        train, val, test = utils.create_fold(df, fold_seed=1234, frac=[0.7, 0.1, 0.2])
    else:
        raise ValueError(f'Unknown model {model_name}')

    return train, val, test
