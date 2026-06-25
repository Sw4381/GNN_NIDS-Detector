# GNN-based Autoencoder for Network Flow Anomaly Detection

This repository contains a refactored version of the GNN-based autoencoder pipeline originally implemented in a Jupyter notebook (`Proposed_LLM.ipynb`). The code provides a **standalone, script-based implementation** for training and evaluating a graph-based anomaly detection model on flow-level network traffic such as CIC-IDS2017.   

The core idea is:

1. **Preprocess raw flow CSVs** into time-windowed data. 
2. **Aggregate per IP-pair features** and build a **temporal multi-digraph** (NetworkX).
3. Convert the multigraph into a **DGL graph** with edge/node features.
4. Train an **enhanced GNN autoencoder + DGI-style objective** to learn normal behavior and detect anomalies via reconstruction error.
5. Evaluate detection performance (AUC, TPR/TNR, confusion matrix, etc.) on a test graph. 

The code is organized as a small Python package so it can be used both as a **command-line script** (`main.py`) and as an **importable module**.

---

## Contents

- `__init__.py`  
  Package entry that re-exports the main functions and models:
  - Preprocessing utilities (`time_window`, scaling helpers)
  - Graph construction (`build_multigraph`, `convert_to_dgl_multigraph`)
  - Models (`EnhancedGNNAutoencoder`, `EnhancedGNNModel`)
  - Training & evaluation helpers (`train_model`, `evaluate`)

- `data_preprocessing.py`  
  - `time_window(df, time_interval="2T")`: converts raw flow CSV into a time-indexed DataFrame with a `"Time Group"` column, after fixing early-morning timestamps.
  - `split_by_time_window(df)`: splits the data into a list of `(time_group, window_df)` tuples.  
  - `train_scaler_on_full_data(flow_data)`: aggregates per-IP-pair features across all windows and trains a global edge feature scaler (StandardScaler). 
  - `apply_scaler_to_features(aggregated_data, scaler)`: applies the learned scaler to a single window’s aggregated features. 
  - `save_scaler`, `load_scaler`, `save_node_scalers`, `load_node_scalers`: utilities to persist/load scalers for edges and node stats.  

- `graph_construction.py`  
  - `extract_ips_from_flow_id` and `extract_features_per_ip_pair(window_data)`: aggregate CIC-IDS-style flow records into **per (Source IP, Destination IP)** features with statistics over packet counts, lengths, and port usage, and assign binary labels. 
  - `extract_node_features(multigraph)`: compute per-node 1-hop structural features such as graph density, out-degree, in-degree, and clustering coefficient.
  - `construct_multigraph_from_features(grouped_features)`: builds a `networkx.MultiDiGraph` with edge attributes including time-group and aggregated features. 
  - `build_multigraph(flow_data, train=True)`: full pipeline that
    - groups data by time
    - aggregates per-IP-pair features (with labels)
    - fits or loads edge and node scalers
    - attaches both scaled and raw features to edges and nodes
    - returns the temporal multigraph and mapping dictionaries. 
  - `convert_to_dgl_multigraph(nx_multigraph, node_to_int_map)`
    - `edata["feat"]`: scaled edge features
    - `edata["raw_feature"]`: raw (unscaled) edge features
    - `edata["time"]`: time-group index
    - `edata["label"]`: binary edge label
    - `ndata["feat"]`: scaled per-node structural features
    - `ndata["node_indicate"]`: simple node indicator feature.

- `gnn_models.py`  
  - `EnhancedGNNAutoencoder`: a two-stage message-passing autoencoder that:
    - projects node and edge features into a shared hidden space,
    - aggregates node messages to edges twice,
    - encodes edge embeddings into a latent space and decodes them back to edge features.
  - `Discriminator`: a simple bilinear discriminator used for DGI-style contrastive learning. 
  - `EnhancedGNNModel`: wraps the autoencoder and discriminator, combining:
    - reconstruction loss on original and augmented edges,
    - DGI-style loss between positive (original/augmented) and negative (corrupted) samples,
    - returns a dictionary of detailed losses and per-edge MSE scores.
    - **Hybrid topology-aware augmentation (Sec. 3.1.3)**: each training view jointly
      applies (i) feature-level Gaussian noise with intensity schedule
      `Lambda = {0.2, 0.4, 0.6}`, (ii) stochastic edge dropping at rate `p_e = 0.05`,
      and (iii) node feature masking at rate `p_n = 0.03`. Controlled by the
      `aug_mode` (`"gaussian" | "edge_drop" | "node_mask" | "hybrid"`, default
      `"hybrid"`), `edge_drop_p`, `node_mask_p`, and `augmentation_levels` arguments.

- `training_eval.py`  
  - `train_model(...)`: training loop that logs epoch-wise loss, measures total training time, and saves the final model checkpoint to disk.
  - `measure_inference_time(...)`: measures average inference time per graph over multiple runs.
  - `evaluate(model, graph, ...)`: computes reconstruction-based anomaly scores, threshold selection, confusion matrix, and standard classification metrics (accuracy, ROC AUC, TPR/TNR/FPR). It also returns detailed detection data (TP, FP, FN, TN sets with raw/original/reconstructed features). :contentReference[oaicite:22]{index=22}  

- `main.py`  
  - Simple **CLI entry point** for training and evaluating `EnhancedGNNModel` on train/test CSVs.

---

## Datasets

The framework is evaluated on five publicly available datasets spanning enterprise,
cloud, industrial IoT, and end-to-end TLS-encrypted DNS-over-HTTPS environments.
All datasets are obtained from their original public sources listed below;
they are **not** redistributed in this repository.

| Dataset | Environment | Source |
|---|---|---|
| **CIC-IDS2017** | On-premises enterprise | https://www.unb.ca/cic/datasets/ids-2017.html |
| **NF-CSE-CIC-IDS2018-v3** | Cloud (NetFlow v3, 53 features) | https://espace.library.uq.edu.au/view/UQ:ece9b83 (UQ NIDS datasets: https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets) |
| **Edge-IIoTset** | Industrial IoT testbed | https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot (IEEE DataPort: https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) |
| **DoH (benign) — Jeřábek corpus** | TLS-encrypted DoH (real-world ISP) | https://zenodo.org/records/5956044 |
| **CIRA-CIC-DoHBrw-2020** | TLS-encrypted DoH (benign + malicious tunneling) | https://www.unb.ca/cic/datasets/dohbrw-2020.html |

**Encrypted DoH Setup 2 / Setup 3** are constructed by combining the Jeřábek
real-world DoH corpus (benign traffic) with the malicious DoH tunneling flows
from CIRA-CIC-DoHBrw-2020 (six tunneling tools: DNS2TCP, DNSCat2, Iodine, dnstt,
tcp-over-dns, tuns). See Section 4.1 of the paper for the exact train/test
temporal-split protocol.

> Please cite the original dataset papers when using them. The intrusion-detection
> benchmarks (CIC-IDS2017, CSE-CIC-IDS2018, DoHBrw-2020) are provided by the
> Canadian Institute for Cybersecurity (UNB); the NetFlow v3 conversion is from
> the University of Queensland NIDS datasets project; Edge-IIoTset is by
> Ferrag et al. (2022); and the real-world DoH corpus is by Jeřábek et al. (2022).

---
