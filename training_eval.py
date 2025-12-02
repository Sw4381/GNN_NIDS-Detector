"""
Refactored modules extracted from `Proposed_LLM.ipynb`.

This file is generated to make the code easier to reuse as a Python module.
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
import dgl
import dgl.function as fn
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_model(model, train_graph, val_graph=None, optimizer=None, lr=0.001, num_epochs=100, 
                device='cpu', save_dir='./models', model_name='gnn_model'):
    """
    Args:
        model: EnhancedGNNModel 또는 EnhancedGNNAutoencoder
        train_graph: 학습 그래프 (DGLGraph)
        val_graph: 검증 그래프 (선택)
        optimizer: 옵티마이저
        lr: 학습률
        num_epochs: 훈련 에폭 수
        device: 'cpu' 또는 'cuda'
        save_dir: 모델 저장 디렉토리
        model_name: 저장할 모델 이름
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    if val_graph is not None:
        val_graph = val_graph.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'val_loss': []
    }

    # 🔹 학습 시작 시간 기록
    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        model.train()
        train_graph = train_graph.to(device)

        output = model(train_graph, include_augmented_recon=True, training=True)
        total_loss = output['total_loss']

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        history['train_loss'].append(total_loss.item())

        log_str = f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {total_loss.item():.4f}"

        if val_graph is not None:
            model.eval()
            with torch.no_grad():
                val_output = model(val_graph, include_augmented_recon=True, training=False)
                val_loss = val_output['total_loss']
                history['val_loss'].append(val_loss.item())
                log_str += f", Val Loss: {val_loss:.4f}"

        print(log_str)

    # 🔹 학습 종료 시간 기록
    end_time = time.perf_counter()
    total_seconds = end_time - start_time
    total_minutes = total_seconds / 60.0
    print(f"[Training] Total time: {total_seconds:.2f} seconds ({total_minutes:.2f} minutes)")

    # 🔹 최종 모델 저장
    final_model_path = os.path.join(save_dir, f"{model_name}_final.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, final_model_path)
    print(f"Final model saved at: {final_model_path}")

    return history, final_model_path, total_seconds





def measure_inference_time(model, graph, device='cpu', n_runs=50, warmup=5):
    model = model.to(device)
    graph = graph.to(device)
    model.eval()

    # 🔹 Warm-up: GPU first-run 오버헤드 제거
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(graph, include_augmented_recon=False, training=False)

    torch.cuda.empty_cache() if device == 'cuda' else None

    # 🔹 실제 측정
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(graph, include_augmented_recon=False, training=False)
    end = time.perf_counter()

    avg_sec = (end - start) / n_runs
    avg_ms = avg_sec * 1000.0
    print(f"[Inference] Avg time per graph: {avg_ms:.4f} ms over {n_runs} runs")
    return avg_sec, avg_ms



from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd



def evaluate(model, graph, find_optimal_threshold=True, return_anomalies=True):
    # 모델이 튜플이나 리스트 형태로 전달되었을 경우 첫 번째 요소를 모델로 간주
    if isinstance(model, (tuple, list)):
        model = model[0]

    model.eval()
    with torch.no_grad():
        # Forward pass - 검증 모드에서는 원본 데이터만 사용 (include_augmented_recon=False, training=False)
        output = model(graph, include_augmented_recon=False, training=False)
        recon_loss = output['reconstruction_loss']

        # Get reconstruction error for each edge
        reconstructed = output['reconstructed_features']
        original = graph.edata["feat"]
        no_scaler_original = graph.edata["raw_feature"]

        # Calculate reconstruction error (anomaly score) for each edge
        # Using mean squared error per edge
        edge_errors = torch.mean((reconstructed - original) ** 2, dim=1)

        # Basic metrics
        mse = F.mse_loss(reconstructed, original)
        mae = F.l1_loss(reconstructed, original)

        print(f"평가 - 재구성 손실: {recon_loss.item():.4f}")
        print(f"평가 - MSE: {mse.item():.4f}, MAE: {mae.item():.4f}")

        results = {
            'reconstruction_loss': recon_loss.item(),
            'mse': mse.item(),
            'mae': mae.item(),
            'embeddings': output['embeddings'],
            'edge_errors': edge_errors
        }

        # Get labels from graph edge features
        if 'label' in graph.edata:
            # Get labels from graph
            labels = graph.edata['label']

            # Convert to numpy for sklearn
            errors_np = edge_errors.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Calculate ROC curve
            fpr_list, tpr_list, thresholds = roc_curve(labels_np, errors_np)
            roc_auc = roc_auc_score(labels_np, errors_np)

            # Find optimal threshold that maximizes ROC AUC
            optimal_idx = np.argmax(tpr_list - fpr_list)
            optimal_threshold = thresholds[optimal_idx]

            if not isinstance(find_optimal_threshold, bool):
                # Use percentile as threshold if a specific value is provided
                optimal_threshold = np.percentile(errors_np, find_optimal_threshold)

            # Make predictions using the optimal threshold
            predictions = (errors_np >= optimal_threshold).astype(int)

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(labels_np, predictions).ravel()

            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


            # Print results
            print(f"\n======= 이상 탐지 평가 지표 =======")
            print(f"• 정확도          : {accuracy:.4f}")
            print(f"• ROC AUC         : {roc_auc:.4f}")
            print(f"• TPR (재현율)    : {tpr:.4f}")
            print(f"• TNR (특이도)    : {tnr:.4f}")
            print(f"• FPR             : {fpr:.4f}")


            # Add metrics to results
            results.update({
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'tpr': tpr,
                'tnr': tnr,
                'fpr': fpr,
                'optimal_threshold': optimal_threshold,
                'confusion_matrix': {
                    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
                }
            })
            print(results['confusion_matrix'])
            # 각 유형별 인덱스 찾기
            tp_indices = np.where((predictions == 1) & (labels_np == 1))[0]  # 정탐(TP): 실제 이상을 이상으로 예측
            fp_indices = np.where((predictions == 1) & (labels_np == 0))[0]  # 오탐(FP): 실제 정상을 이상으로 예측
            fn_indices = np.where((predictions == 0) & (labels_np == 1))[0]  # 미탐(FN): 실제 이상을 정상으로 예측
            tn_indices = np.where((predictions == 0) & (labels_np == 0))[0]  # 정상 정확(TN): 실제 정상을 정상으로 예측

            # Return detected anomalies if requested
            if return_anomalies:
                # 모든 이상 예측(TP+FP)에 대한 정보
                anomaly_indices = np.where(predictions == 1)[0]
                # 각 유형별 데이터 준비
                def prepare_data_for_indices(indices):
                    if len(indices) == 0:
                        return None
                    return {
                        'indices': indices,
                        'errors': errors_np[indices],
                        'edge_features': original[indices].cpu().numpy(),
                        'raw_edge_features': no_scaler_original[indices].cpu().numpy(),
                        'reconstructed_features': reconstructed[indices].cpu().numpy(),
                        'actual_labels': labels_np[indices]
                    }

                # Get original edge features and other relevant information for all categories
                detection_data = {
                    'all_anomalies': {  # 모든 이상 예측(TP+FP)
                        'indices': anomaly_indices,
                        'errors': errors_np[anomaly_indices],
                        'edge_features': original[anomaly_indices].cpu().numpy(),
                        'raw_edge_features': no_scaler_original[anomaly_indices].cpu().numpy(),
                        'reconstructed_features': reconstructed[anomaly_indices].cpu().numpy(),
                        'actual_labels': labels_np[anomaly_indices]
                    },
                    'true_positives': prepare_data_for_indices(tp_indices),  # 정탐(TP) # 악-> 악
                    'false_positives': prepare_data_for_indices(fp_indices),  # 오탐(FP) # 정-> 악
                    'false_negatives': prepare_data_for_indices(fn_indices),  # 미탐(FN)
                    'true_negatives': prepare_data_for_indices(tn_indices),  # 정상 정확(TN)
                }

                # Add detection data to results
                results['detection_data'] = detection_data

        return results