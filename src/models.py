import os

from DeepPurpose import utils, CompoundPred

from .tree_model import TreeModel


def get_model(model_name, **model_kwargs):
    if model_name in ['Transformer', 'DGL_GCN']:
        config = utils.generate_config(drug_encoding=model_name, **model_kwargs)
        model = CompoundPred.model_initialize(**config)
        return model
    elif model_name == 'tree':
        model = TreeModel(**model_kwargs)
        return model
    else:
        raise ValueError(f"Unknown model {model_name}")
        
        
