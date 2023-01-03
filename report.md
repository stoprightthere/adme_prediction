# Report

Three publically available datasets were chosen to benchmark graph neural networks, transformer models, and classical tree models for ADME properties prediction:
- HIA (Human Intestinal Absorption), Hou et al.
- BBB (Blood-Brain Barrier), Martins et al.
- CYP3A4 Substrate, Carbon-Mangels et al.

Three respective models were trained on these datasets. For each dataset, 70% of data were used for training, 10% for validation, and 20% were held out for benchmarking.

Trained models were evaluated on held-out datasets via AUC-ROC score and F1 score. Then, harmonic mean over these datasets was computed for each dataset.

|             | Mean AUC |  Mean F1 |
|:------------|---------:|---------:|
| Tree model  | 0.753438 | 0.813756 |
| Graph NN    | 0.824711 | 0.838888 |
| Transformer | 0.711308 | 0.815738 |

The higher the value, the better is the model's performance. As is seen, Graph NN model consistently outperforms both the classical tree model and the transformer model.
