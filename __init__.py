"""
Top-level package for the refactored GNN + LLM pipeline.

This package was extracted from `Proposed_LLM.ipynb` to make the code easier to
reuse as a standalone Python module.

NOTE:
- Some higher-level analysis helpers such as `EdgeTraceMapper`,
  `StatisticalAttackProfiler`, and `BehavioralRoleProfiler` are referenced
  in `flow_analysis.py` but not defined in the original notebook. You will
  need to provide their implementations (or stub versions) if you want to
  run the full LLM-based explanation pipeline end-to-end.
"""

from .data_preprocessing import (
    time_window,
    save_scaler,
    load_scaler,
    save_node_scalers,
    load_node_scalers,
    train_scaler_on_full_data,
    apply_scaler_to_features,
    split_by_time_window,
)

from .graph_construction import (
    extract_ips_from_flow_id,
    extract_features_per_ip_pair,
    extract_node_features,
    construct_multigraph_from_features,
    build_multigraph,
    convert_to_dgl_multigraph,
)

from .gnn_models import EnhancedGNNAutoencoder, Discriminator, EnhancedGNNModel
from .training_eval import train_model, measure_inference_time, evaluate
from .flow_analysis import run_full_flow_analysis
