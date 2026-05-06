# IEEE-CIS Fraud Detection: Classical ML vs Graph Neural Networks
A comparative study of imbalance handling techniques for credit card fraud detection, evaluating six classical machine learning baselines and four Graph Neural Network architectures on the full IEEE-CIS dataset (590,540 transactions, 3.50% fraud rate).
Best result: GraphSAGE with NeighborLoader sampling. AUC-ROC 0.894, AUC-PR 0.523, F1 0.519, Recall@1% 0.869
Built for CA683 Data Mining at Dublin City University (April 2026). Full methodology and findings are documented in the accompanying IEEE-format report.

## Problem
Credit card fraud detection is a severely imbalanced classification problem. Only 3.5% of transactions in the IEEE-CIS dataset are fraudulent. Standard classifiers default to predicting the majority class, achieving high accuracy but near-zero fraud recall. We investigate which imbalance handling techniques are most effective, and whether Graph Neural Networks leveraging transaction relationships can outperform tabular baselines.

## Research Questions
RQ1. Which imbalance handling technique, SMOTE, class weighting, or cost-sensitive parameters, is most effective for classical models on high-dimensional transaction data?
RQ2. Do GNNs, leveraging graph structure, outperform classical baselines on precision-recall metrics?
RQ3. Does NeighborLoader mini-batch training improve GNN performance over full-batch training?

## Methodology
We followed the CRISP-DM framework end-to-end. Stratified 70/15/15 train/validation/test split (413,614 / 88,345 / 88,581 rows). Classification thresholds were optimised on the validation set across 98 candidate values (0.01 to 0.99) and final metrics were computed exclusively on the held-out test set.
### Classical Baselines (six models)

Logistic Regression with class_weight='balanced' and SMOTE
Random Forest with class_weight='balanced' and SMOTE
LightGBM with scale_pos_weight set to the imbalance ratio (27.6)
Isolation Forest with contamination set to the true fraud rate (0.035)

### Graph Construction
Transactions modelled as nodes, with edges connecting transactions sharing any of seven attributes (card1, card2, addr1, addr2, P_emaildomain, R_emaildomain, DeviceInfo). Groups larger than 50 transactions sharing the same attribute were excluded to prevent super-nodes from dominating message-passing.
Final graph: 590,540 nodes, 2,091,012 edges, average degree 3.54.
### GNN Architectures (four models)

GCN: spectral convolution with equal-weight neighbour aggregation
GAT: multi-head attention (4 heads layer 1, 1 head layer 2)
GraphSAGE: inductive mean aggregation with skip-connection-style concatenation
CS-GraphSAGE: GraphSAGE backbone with a learnable 2x2 cost matrix initialised at C[1,0]=28 (the imbalance ratio), adapted from Hu et al. (2024)

All GNNs trained with PyTorch Geometric's NeighborLoader (2-hop sampling, 15+10 neighbours per hop, batch size 2048). Hidden dimension 128, dropout 0.3, AdamW with weight decay 5e-4, ReduceLROnPlateau scheduler, early stopping on validation AUC-ROC with patience 15.

## Results
| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall@1% |
| --- | --- | --- | --- | --- | --- |
| GraphSAGE (NeighborLoader) | 0.894 | 0.523 | 0.519 | 0.602 | 0.869 |
| RF (Class Weighted) | 0.885 | 0.518 | 0.507 | 0.596 | 0.880 |
| CS-GraphSAGE | 0.880 | 0.450 | 0.462 | 0.519 | 0.763 |
| GCN | 0.877 | 0.474 | 0.479 | 0.518 | 0.819 |
| LightGBM | 0.873 | 0.344 | 0.418 | 0.375 | 0.598 |
| RF (SMOTE) | 0.864 | 0.455 | 0.455 | 0.551 | 0.837 |
| GAT | 0.842 | 0.328 | 0.361 | 0.342 | 0.589 |
| LR (Class Weighted) | 0.838 | 0.263 | 0.330 | 0.297 | 0.514 |
| LR (SMOTE) | 0.830 | 0.246 | 0.321 | 0.289 | 0.485 |
| Isolation Forest | 0.751 | 0.096 | 0.202 | 0.132 | 0.061 |
### NeighborLoader vs Full-Batch Training (RQ3)
| GNN | Full-Batch AUC-PR | NeighborLoader AUC-PR | Gain |
| --- | --- | --- | --- |
| GCN | 0.338 | 0.474 | +40.2% |
| GAT | 0.259 | 0.328 | +26.6% |
| GraphSAGE | 0.389 | 0.523 | +34.4% |
| CS-GraphSAGE | 0.413 | 0.450 | +9.0% |

NeighborLoader improves every GNN substantially. We propose two mechanisms: mini-batch stochasticity acts as implicit regularisation, and per-batch neighbourhood sampling concentrates fraud signal that gets diluted under full-batch propagation.

## Key Findings
RQ1: Class weighting beats SMOTE for tree-based models. RF (Class Weighted) achieves AUC-PR 0.518 vs RF (SMOTE) at 0.455, a 14% gap on the most informative metric for imbalanced classification. SMOTE adds synthetic interpolations in 432-dimensional space that don't reflect realistic fraud patterns, while class weighting preserves the original feature distribution.
RQ2: GNNs outperform classical baselines on precision-recall. GraphSAGE achieves the highest AUC-PR (0.523) and AUC-ROC (0.894) across all ten models, narrowly beating RF (Class Weighted) on AUC-PR while improving F1 by 1.2 points. GCN (0.474 AUC-PR) also beats LightGBM (0.344). GAT underperforms, consistent with Liu et al. (2021), who observed attention mechanisms struggle when the minority class lacks sufficient labelled neighbourhood signal.
RQ3: NeighborLoader is the single most impactful design choice for GNNs, improving AUC-PR by 9% to 40% across all four architectures.
Threshold optimisation matters enormously. With a fixed 0.5 threshold, LightGBM yields F1 = 0.000 because calibrated probabilities at a 3.5% fraud rate rarely exceed 0.5. Validation-tuned thresholds (range 0.11 to 0.86 across models) were essential to recover usable F1 scores.
CS-GraphSAGE underperformed plain GraphSAGE despite the learnable cost matrix. In the original CSGNN paper (Hu et al. 2024), the cost matrix is paired with RL-controlled neighbourhood sampling. Our NeighborLoader uses uniform sampling instead. Without the targeted positive-neighbour selection, the cost matrix alone wasn't enough.

## Tech Stack

Frameworks: PyTorch 2.10, PyTorch Geometric 2.7, torch-sparse, scikit-learn, LightGBM, imbalanced-learn
Data: pandas, NumPy
Environment: Kaggle Notebooks, T4 GPU (15.6 GB VRAM, CUDA 12.8), 30 GB RAM


## Repository Structure

fraud_detection_pipeline.ipynb: Main notebook with full pipeline (baselines + GNNs + results)
report.pdf: IEEE-format research report (full methodology, related work, discussion)
README.md: This file
requirements.txt: Python dependencies


## How to Run
The notebook is written for Kaggle Notebooks (free T4 GPU, dataset attached automatically) but runs in any environment with a CUDA-capable GPU.
On Kaggle:

Fork the IEEE-CIS Fraud Detection competition data into your workspace
Upload the notebook and attach the dataset
Settings, Accelerator, GPU T4 x2
Run all cells

Locally: install requirements.txt, download the IEEE-CIS dataset from Kaggle, update the DATA_PATH variable in Step 2, then run.
Full pipeline runtime is approximately 2.5 hours on a single T4 (most of it spent on GCN training).

## References (selected)
The full bibliography is in report.pdf. Key sources:

Liu et al. (2021), Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection, ACM Web Conf.
Hu et al. (2024), Cost-Sensitive GNN-Based Imbalanced Learning for Mobile Social Network Fraud Detection, IEEE Trans. Comput. Social Syst.
Motie & Raahemi (2024), Financial Fraud Detection Using Graph Neural Networks: A Systematic Review, Expert Syst. Appl.
Baisholan et al. (2025), A Systematic Review of Machine Learning in Credit Card Fraud Detection Under Original Class Imbalance, Computers.


## Authors
A group project for CA683 Data Mining at Dublin City University, April 2026.

Tharakesh Aravindan S.T: LinkedIn
Eoin Delhunty
Suman Neupane
Rehoboth Salako
