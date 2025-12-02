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


def time_window(flow_data, time_interval="2T"):
    """
    네트워크 트래픽 데이터를 주어진 시간 간격(time_interval)별로 그룹화하여 'Time Group' 컬럼 추가.
    """
    flow_data = flow_data.copy()  # 원본 손상 방지
    if "Timestamp" not in flow_data.columns:
        raise ValueError("DataFrame에 'Timestamp' 컬럼이 없습니다.")

    flow_data["Timestamp"] = pd.to_datetime(flow_data["Timestamp"], errors='coerce')
    flow_data = flow_data.dropna(subset=["Timestamp"])  # Timestamp 변환 실패한 행 제거

    # 새벽 1시~6시 데이터를 12시간 앞으로 이동
    flow_data["Timestamp"] = flow_data["Timestamp"].apply(
        lambda x: x + pd.Timedelta(hours=12) if 1 <= x.hour <= 6 else x
    )

    flow_data = flow_data.sort_values("Timestamp").reset_index(drop=True)

    # 2분 간격으로 그룹화
    flow_data['Time Group'] = flow_data['Timestamp'].dt.floor(time_interval)

    return flow_data



def save_scaler(scaler, filename="edge_scaler.pkl"):
    """
    스케일러 객체를 파일로 저장
    """
    with open(filename, 'wb') as f:
        pickle.dump(scaler, f)



def load_scaler(filename="edge_scaler.pkl"):
    """
    스케일러 객체를 파일에서 로드
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)



def save_node_scalers(scalers, filename="node_scalers.pkl"):
    """
    노드 특성 스케일러 사전을 파일로 저장
    """
    with open(filename, 'wb') as f:
        pickle.dump(scalers, f)



def load_node_scalers(filename="node_scalers.pkl"):
    """
    노드 특성 스케일러 사전을 파일에서 로드
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)




def train_scaler_on_full_data(flow_data):
    """
    전체 데이터 프레임을 기준으로 엣지 Scaler를 학습하여 저장
    """
    from graph_construction import extract_features_per_ip_pair
    all_features = []
    grouped_data = split_by_time_window(flow_data)

    for _, window_data in grouped_data:
        aggregated_data = extract_features_per_ip_pair(window_data)
        if aggregated_data is not None:
            all_features.append(aggregated_data)

    full_data = pd.concat(all_features, ignore_index=True)
    feature_columns = [col for col in full_data.columns if col not in ['ip_pair', 'Label', 'time_group']]

    scaler = StandardScaler()
    scaler.fit(full_data[feature_columns])
    save_scaler(scaler)

    return scaler



def apply_scaler_to_features(aggregated_data, scaler):
    """
    개별 윈도우 데이터에 대해 학습된 엣지 Scaler 적용
    """
    feature_columns = [col for col in aggregated_data.columns if col not in ['ip_pair', 'Label', 'time_group']]
    aggregated_data[feature_columns] = scaler.transform(aggregated_data[feature_columns])
    return aggregated_data




def split_by_time_window(flow_data):
    # 시간대별 데이터 그룹화하여 리스트 반환
    grouped_data = [(time_group, window_data) for time_group, window_data in flow_data.groupby("Time Group")]

    return grouped_data

