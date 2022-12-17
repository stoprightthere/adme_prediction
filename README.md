# ADME properties prediction

A quick benchmark of GNN, Transformer and Decision tree models for ADME properties prediction.

## Local setup
To run locally, clone the repository and create a `conda` environment from `environment.yml`

The main jupyter notebook is `compare_models_locally.ipynb` and the findings a summarized in a quick `report.md` report.

## What was done
After a bit of googling, I found out about `DeepPurpose` library that had graph neural net and transformer models implemented. `tdc` library supports  `DeepPurpose`'s data format. So, for these models, I am using `DeepPurpose`.

Also, I did a little literature search and found a paper [1] with [code](https://github.com/smu-tao-group/ADMET_XGBoost).

In `quick_dirty.ipynb` notebook I implemented a quick and dirty training of a model on the 'HIA' dataset just to see what is going on. Turned out the model trained quite quickly, so I did not need to use cloud for the problem, I could train locally.

In `explore_locally.ipynb` notebook there is a quick dataset analysis. I chose three datasets 
`HIA_Hou`, `BBB_Martins`, `CYP3A4_Substrate_CarbonMangels` because:
- these all pertain to ADME property prediction
- they are quite small
- they are all binary classification datasets
- randomly

So, for the sake of simplicity, I constrained myself to these datasets. It would not be a problem to include more (or all) datasets, but it would be more hassle.

The datasets are clean but the labels are quite imbalanced. Hence, I chose AUC-ROC and F1 scores as performance metrics. AUC-ROC is also used in the paper [1]. Besides, a sales team probably needs some kind of a cumulative metric, not a table with lots of datasets and lots of metrics. So, I average across the datasets using harmonic mean. 

Tree model is not implemented in DeepPurpose, but `sklearn` is the place to go. By a "classical tree model", I understand a literal decision tree. I wrote a wrapper around `DecisionTreeClassifier` so that it reminds `DeepPurpose`'s models. Also, one has to transform drug `SMILES` to features. The paper [1] uses a bunch of transforms. I use only `Pubchem` transform from `DeepPurpose` for simpliticity. It would not be hard to use other transforms. It would also be relatively easy to use other models (like `XGBoost`), if the sales team needs.

Then, the models are trained, AUC-ROC and F1 score are computed, and their harmonic mean accross datasets is calculated to arrive to a cumulative metric of model performance.
For the sake of simplicity, neural network models are trained for 100 epochs, with a batch size of 64, and the tree has the maximum depth of 5. The hyperparameters were chosen ad-hoc.

The problem took me basically a day. 

## What could have been done
It would be better to train for more epochs and choose hyperparameters in a principled way, like via cross-validation or using Bayesian hyperparameter tuning.

One could also include regression-type datasets. It would only require small code modifications (like, using a tree regressor instead of a classifier) and using appropriate metrics. 

That said, the goal was not to actually train the models but rather set up an easily-extensible pipeline.

## References
[1] [Accurate ADMET Prediction with XGBoost](https://arxiv.org/pdf/2204.07532.pdf).
