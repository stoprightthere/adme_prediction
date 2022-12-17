import os

from DeepPurpose import utils, CompoundPred

from .tree_model import TreeModel, model_pretrained


_DEEPPURPOSE_MODELS = ['Transformer', 'DGL_GCN']


def get_model(model_name, **model_kwargs):
    """
    Get a model to be trained. 

    :param model_name: either 'Transformer', 'DGL_GCN' (Graph Convolution Network) or 'tree' (decistion tree model).
    :param **model_kwargs: model parameters.

    :return: model.
    """
    if model_name in _DEEPPURPOSE_MODELS:
        config = utils.generate_config(drug_encoding=model_name, **model_kwargs)
        model = CompoundPred.model_initialize(**config)
        return model
    elif model_name == 'tree':
        model = TreeModel(**model_kwargs)
        return model
    else:
        raise ValueError(f'Unknown model {model_name}')
        
        
def load_pretrained(model_name, model_dir):
    """
    Load pretrained model from disk.

    :param model_name:  either 'Transformer', 'DGL_GCN' (Graph Convolution Network) or 'tree' (decistion tree model).
    :param model_dir: the directory where the model has been saved.

    :return: model.
    """
    if model_name in _DEEPPURPOSE_MODELS:
        model = CompoundPred.model_pretrained(model_dir)
    elif model_name == 'tree':
        model = model_pretrained(model_dir)
    else:
        raise ValueError(f'Unknown model {model_name}')
    return model
                             
