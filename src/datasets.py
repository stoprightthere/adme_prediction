import pandas as pd

from DeepPurpose import utils
from tdc.single_pred import ADME


def prepare_dataset(dataset_name, model_name):
    data = ADME(name=dataset_name)
    X, y = data.get_data(format = 'DeepPurpose')

    if model_name in ['Transformer', 'DGL_GCN']:
        train, val, test = utils.data_process(
            X_drug = X, 
            y = y, 
            drug_encoding = model_name,
            random_seed = 'TDC',
        )
    elif model == 'tree':
        df = pd.DataFrame(zip(X, y))
        df.rename(columns={0: 'SMILES', 1: 'Label'}, inplace=True)
        df = utils.encode_drug(df, 'Pubchem')
        train, val, test = utils.create_fold(df, random_seed=1234, frac=[0.7, 0.1, 0.2])
    else:
        raise ValueError(f'Unknown model {model_name}')

    return train, val, test
