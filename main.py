"""
Run from the Code directory, e.g.:

    python3 main.py \
      --train-csv "/home/user/Sunwoo/SCI 코드/Monday-WorkingHours.pcap_ISCX.csv" \
      --test-csv  "/home/user/Sunwoo/SCI 코드/dataset_2/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
"""

import argparse
import torch
import pandas as pd

from data_preprocessing import time_window
from graph_construction import build_multigraph, convert_to_dgl_multigraph
from gnn_models import EnhancedGNNModel
from training_eval import train_model, evaluate


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate EnhancedGNNModel.")
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to training CSV (e.g., Monday data).",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="Path to test CSV (e.g., Tuesday/Friday data).",
    )
    parser.add_argument(
        "--time-interval",
        type=str,
        default="2min",  # '2T' 대신 '2min' 사용 (FutureWarning 방지)
        help="Pandas time interval (e.g., '2min').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    args = parser.parse_args()

    # 1. Load & group data
    print(f"[INFO] Loading training CSV: {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)

    print(f"[INFO] Loading test CSV: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)

    print(f"[INFO] Applying time windowing with interval = {args.time_interval}")
    train_grouped = time_window(train_df, time_interval=args.time_interval)
    test_grouped = time_window(test_df, time_interval=args.time_interval)

    # 여기서는 단순화를 위해 첫 번째 타임 윈도우만 사용
    print("[INFO] Building graphs from grouped data...")
    train_multigraph, node_to_int, int_to_node, int_to_time, _ = build_multigraph(
        train_grouped, train=True
    )
    train_dgl_graph = convert_to_dgl_multigraph(train_multigraph, node_to_int)

    test_multigraph, test_node_to_int, test_int_to_node, test_int_to_time, _ = build_multigraph(
        test_grouped, train=False
    )
    test_dgl_graph = convert_to_dgl_multigraph(test_multigraph, test_node_to_int)

    # 2. Instantiate model
    edge_dim = train_dgl_graph.edata["feat"].shape[1]
    node_dim = train_dgl_graph.ndata["feat"].shape[1]
    print(f"[INFO] edge_dim = {edge_dim}, node_dim = {node_dim}")

    model = EnhancedGNNModel(edge_dim=edge_dim, node_dim=node_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    # model.to(device)

    # 3. Train
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[INFO] Start training for {args.epochs} epochs...")
    history = train_model(
        model,
        train_graph=train_dgl_graph,
        val_graph=None,
        optimizer=optimizer,
        lr=args.lr,
        num_epochs=args.epochs
        # device=device,
    )

    # 4. Evaluate on test graph
    print("[INFO] Evaluating on test graph...")
    results = evaluate(model, test_dgl_graph)

if __name__ == "__main__":
    main()
