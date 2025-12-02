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
from data_preprocessing import *


def extract_ips_from_flow_id(flow_id):
    parts = flow_id.split('-')
    if len(parts) >= 2:  # 최소한 Source IP와 Destination IP가 있는지 확인
        source_ip = parts[0]
        destination_ip = parts[1]
        return tuple([source_ip, destination_ip])
    return None

def extract_features_per_ip_pair(window_data):
    """
    특정 시간대의 네트워크 트래픽 데이터에서 IP-Pair 별 Feature 추출 및 Label 설정.
    """
    if window_data.empty:
        return None


    # window_data['ip_pair'] = window_data['Flow ID'].apply(extract_ips_from_flow_id)
    window_data['ip_pair'] = window_data.apply(lambda x: tuple([x['Source IP'], x['Destination IP']]), axis=1)
    window_data['Label_Binary'] = window_data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    ip_pair_labels = window_data.groupby('ip_pair')['Label_Binary'].max().reset_index()
    ip_pair_labels.rename(columns={'Label_Binary': 'Label'}, inplace=True)

    aggregated_data = window_data.groupby('ip_pair').agg(
        Total_Fwd_Packets_mean=('Total Fwd Packets', 'mean'),
        Total_Fwd_Packets_max=('Total Fwd Packets', 'max'),
        Total_Fwd_Packets_min=('Total Fwd Packets', 'min'),
        Total_Fwd_Packets_std=('Total Fwd Packets', 'std'),

        Total_Bwd_Packets_mean=('Total Backward Packets', 'mean'),
        Total_Bwd_Packets_max=('Total Backward Packets', 'max'),
        Total_Bwd_Packets_min=('Total Backward Packets', 'min'),
        Total_Bwd_Packets_std=('Total Backward Packets', 'std'),

        Total_Length_Fwd_mean=('Total Length of Fwd Packets', 'mean'),
        Total_Length_Fwd_max=('Total Length of Fwd Packets', 'max'),
        Total_Length_Fwd_min=('Total Length of Fwd Packets', 'min'),
        Total_Length_Fwd_std=('Total Length of Fwd Packets', 'std'),

        Total_Length_Bwd_mean=('Total Length of Bwd Packets', 'mean'),
        Total_Length_Bwd_max=('Total Length of Bwd Packets', 'max'),
        Total_Length_Bwd_min=('Total Length of Bwd Packets', 'min'),
        Total_Length_Bwd_std=('Total Length of Bwd Packets', 'std'),

        Well_Known_Dst_Ports_Count=('Destination Port', lambda x: len([p for p in list(x) if 0 <= p <= 1024])),
        Well_Known_Dst_Ports_Unique=('Destination Port', lambda x: len(set([p for p in list(x) if 0 <= p <= 1024]))),
        Total_Dst_Ports_Unique=('Destination Port', lambda x: len(set(x))),

        Well_Known_Src_Ports_Count=('Source Port', lambda x: len([p for p in list(x) if 0 <= p <= 1024])),
        Well_Known_Src_Ports_Unique=('Source Port', lambda x: len(set([p for p in list(x) if 0 <= p <= 1024]))),
        Total_Src_Ports_Unique=('Source Port', lambda x: len(set(x))),

        Num_of_Flows=('Source IP', 'count')
    ).reset_index()

    aggregated_data['Well_Known_Dst_Ports_Unique_Ratio'] = aggregated_data['Well_Known_Dst_Ports_Unique'] / aggregated_data['Total_Dst_Ports_Unique']
    aggregated_data['Well_Known_Src_Ports_Unique_Ratio'] = aggregated_data['Well_Known_Src_Ports_Unique'] / aggregated_data['Total_Src_Ports_Unique']
    aggregated_data = aggregated_data.drop(columns=["Total_Dst_Ports_Unique", "Total_Src_Ports_Unique"])

    aggregated_data = aggregated_data.merge(ip_pair_labels, on="ip_pair", how="left")
    aggregated_data = aggregated_data.fillna(0)

    return aggregated_data



def extract_node_features(multigraph):
    """
    각 노드의 1홉 서브그래프에서 특성을 추출하는 함수

    Args:
        multigraph: NetworkX 멀티그래프

    Returns:
        node_features: 각 노드별 특성 사전
    """
    node_features = {}

    for node in multigraph.nodes():
        # 1-홉 이웃 노드 추출 (나가는 방향과 들어오는 방향 모두 포함)
        neighbors = set(multigraph.successors(node)) | set(multigraph.predecessors(node))

        if not neighbors:  # 이웃이 없는 경우 처리
            node_features[node] = {
                'graph_density': 0,
                'source_out_degree': 0,
                'target_in_degree': 0,
                'clustering_coefficient': 0
            }
            continue

        # 1-홉 서브그래프 추출
        one_hop_subgraph = multigraph.subgraph([node] + list(neighbors))

        # 그래프 밀도 계산
        n_nodes = one_hop_subgraph.number_of_nodes()
        n_edges = one_hop_subgraph.number_of_edges()
        max_possible_edges = n_nodes * (n_nodes - 1)  # 방향이 있는 그래프이므로
        graph_density = n_edges / max_possible_edges if max_possible_edges > 0 else 0

        # 소스 노드의 Out-Degree
        source_out_degree = multigraph.out_degree(node)

        # 타겟 노드의 In-Degree
        target_in_degree = multigraph.in_degree(node)

        # 클러스터링 계수 (NetworkX의 함수 사용)
        # 방향성 그래프를 무방향 그래프로 간주하여 계산
        undirected_subgraph = nx.Graph(one_hop_subgraph)
        try:
            # 노드가 한 개인 경우 예외 처리
            clustering_coefficient = nx.clustering(undirected_subgraph, node)
        except:
            clustering_coefficient = 0

        node_features[node] = {
            'graph_density': graph_density,
            'source_out_degree': source_out_degree,
            'target_in_degree': target_in_degree,
            'clustering_coefficient': clustering_coefficient
        }

    return node_features




def construct_multigraph_from_features(grouped_features):
    """
    시간대별 IP-Pair Feature를 하나의 멀티그래프로 통합하고
    time_group 인덱스 번호와 실제 시간을 매핑하는 map을 생성

    Args:
        grouped_features (list): (time_group, aggregated_data) 튜플 리스트

    Returns:
        G (nx.MultiDiGraph): 생성된 멀티디렉티드 그래프
        int_to_time_map (dict): 인덱스 -> 실제 시간 매핑 딕셔너리
    """
    G = nx.MultiDiGraph()
    int_to_time_map = {}

    for idx, (time_group, aggregated_data) in enumerate(grouped_features):
        int_to_time_map[idx] = time_group  # 인덱스를 타임 그룹에 매핑
        for _, row in aggregated_data.iterrows():
            src, dst = row['ip_pair']
            edge_attr = row.to_dict()
            edge_attr['time_group'] = idx  # 인덱스 번호를 time_group으로 저장
            G.add_edge(src, dst, **edge_attr)

    return G, int_to_time_map




def build_multigraph(flow_data, train=True):
    """
    전체 프로세스를 실행하여 노드와 엣지 특성이 모두 포함된 멀티그래프 생성
    """
    grouped_data = split_by_time_window(flow_data)

    # 전체 노드 집합 생성 및 매핑
    all_nodes = set()
    for _, window_data in grouped_data:
        all_nodes.update(window_data['Source IP'])
        all_nodes.update(window_data['Destination IP'])

    node_to_int_map = {ip: idx for idx, ip in enumerate(sorted(all_nodes))}
    int_to_node_map = {idx: ip for ip, idx in node_to_int_map.items()}

    # 엣지 스케일러 학습 또는 로드
    if train:
        edge_scaler = train_scaler_on_full_data(flow_data)
    else:
        edge_scaler = load_scaler()

    # 시간대별 Feature 추출 및 스케일링
    grouped_features = []
    no_scale_grouped_features = []

    for time_group, window_data in grouped_data:
        aggregated_data = extract_features_per_ip_pair(window_data)
        if aggregated_data is not None:
            # 원본 데이터 저장 (시간 그룹 정보 포함)
            no_scale_grouped_features.append((time_group, aggregated_data.copy()))

            # 스케일된 데이터 저장
            scaled_data = apply_scaler_to_features(aggregated_data, edge_scaler)
            grouped_features.append((time_group, scaled_data))

    # 멀티그래프 생성
    multigraph,int_to_time_map = construct_multigraph_from_features(grouped_features)

    # 원본 특성을 멀티그래프에 추가
    raw_features_map = {}

    # 원본 특성 맵 생성
    for time_group, data in no_scale_grouped_features:
        for _, row in data.iterrows():
            src, dst = row['ip_pair']
            key = (src, dst, time_group)  # 유니크한 키 생성
            raw_features_map[key] = row.drop('ip_pair').to_dict()

    # 그래프의 각 엣지에 원본 특성 추가
    for u, v, k, attr in multigraph.edges(data=True, keys=True):
        time_group = attr.get('time_group')
        key = (u, v, time_group)
        if key in raw_features_map:
            # 'raw_' 접두사를 붙여 원본 특성 추가
            raw_attrs = {'raw_' + k: v for k, v in raw_features_map[key].items()}
            # 기존 엣지 속성에 원본 특성 속성 업데이트
            nx.set_edge_attributes(multigraph, {(u, v, k): raw_attrs})

    # 노드 특성 추출
    node_features = extract_node_features(multigraph)

    # 노드 특성 스케일링을 위한 데이터 수집
    feature_keys = ['graph_density', 'source_out_degree', 'target_in_degree', 'clustering_coefficient']
    node_features_data = {key: [] for key in feature_keys}

    for node, features in node_features.items():
        for key in feature_keys:
            node_features_data[key].append(features[key])

    # 노드 특성 스케일러 학습 또는 로드
    if train:
        node_scalers = {}
        for key in feature_keys:
            values = np.array(node_features_data[key]).reshape(-1, 1)
            scaler = StandardScaler()
            scaler.fit(values)
            node_scalers[key] = scaler
        save_node_scalers(node_scalers)
    else:
        node_scalers = load_node_scalers()

    # 노드 특성 스케일링 및 그래프에 추가
    for node, features in node_features.items():
        # 원본 특성 추가
        nx.set_node_attributes(multigraph, {node: features})

        # 스케일링된 특성 추가
        scaled_features = {}
        for key in feature_keys:
            value = np.array([[features[key]]])
            scaled_value = node_scalers[key].transform(value)[0][0]
            scaled_features[key] = scaled_value

        nx.set_node_attributes(multigraph, {node: scaled_features}, "scaled")

    return multigraph, node_to_int_map, int_to_node_map,int_to_time_map,grouped_features




def convert_to_dgl_multigraph(nx_multigraph, node_to_int_map):
    """
    NetworkX 멀티그래프를 DGL 그래프로 변환 (노드 특성 포함)
    """
    src_nodes, dst_nodes = [], []
    edge_features = []
    raw_edge_features = []  # 원본(스케일되지 않은) 특성 저장
    edge_time_features = []
    labels = []

    # 모든 에지에 대해 처리
    for u, v, key, attr in nx_multigraph.edges(data=True, keys=True):
        src = node_to_int_map[u]
        dst = node_to_int_map[v]
        src_nodes.append(src)
        dst_nodes.append(dst)

        # 스케일된 특성과 원본 특성 분리
        feature_list = []
        raw_feature_list = []

        for feat_key, value in attr.items():
            if feat_key == 'ip_pair':
                continue
            elif feat_key == 'Label':
                labels.append(value)
            elif feat_key == 'time_group':
                edge_time_features.append(value)
            elif feat_key.startswith('raw_'):
                # 'raw_' 접두사가 있는 feature는 원본 특성 리스트에 추가
                raw_feat_key = feat_key[4:]  # 'raw_' 접두사 제거
                raw_feature_list.append(value)
            else:
                # 일반 feature는 스케일된 특성 리스트에 추가
                feature_list.append(value)

        edge_features.append(feature_list)
        raw_edge_features.append(raw_feature_list)

    # DGL 그래프 생성
    dgl_graph = dgl.graph((torch.tensor(src_nodes), torch.tensor(dst_nodes)), 
                         num_nodes=len(node_to_int_map))

    # 엣지 Feature 추가
    dgl_graph.edata['feat'] = torch.tensor(edge_features, dtype=torch.float32)
    dgl_graph.edata['raw_feature'] = torch.tensor(raw_edge_features, dtype=torch.float32)  # 원본 특성 추가
    dgl_graph.edata['time'] = torch.tensor(edge_time_features, dtype=torch.float32).view(-1, 1)
    dgl_graph.edata['label'] = torch.tensor(labels, dtype=torch.float32)

    # 노드 Feature 추가 (노드 특성 정보 포함)
    node_features_list = []
    for node_idx in range(len(node_to_int_map)):
        # node_idx에 해당하는 원래 노드 ID 찾기
        original_node_id = list(node_to_int_map.keys())[list(node_to_int_map.values()).index(node_idx)]

        # 노드 특성 가져오기 (없으면 기본값 사용)
        node_data = nx_multigraph.nodes[original_node_id]
        if node_data:
            # 스케일링된 특성 사용 (있는 경우)
            if 'scaled' in node_data:
                feature_vector = [
                    node_data['scaled'].get('graph_density', 0),
                    node_data['scaled'].get('source_out_degree', 0),
                    node_data['scaled'].get('target_in_degree', 0),
                    node_data['scaled'].get('clustering_coefficient', 0)
                ]
            else:
                # 원본 특성 사용
                feature_vector = [
                    node_data.get('graph_density', 0),
                    node_data.get('source_out_degree', 0),
                    node_data.get('target_in_degree', 0),
                    node_data.get('clustering_coefficient', 0)
                ]
        else:
            # 특성이 없는 경우 0으로 채움
            feature_vector = [0, 0, 0, 0]

        node_features_list.append(feature_vector)

    # 노드 특성을 텐서로 변환하여 그래프에 추가
    dgl_graph.ndata['feat'] = torch.tensor(node_features_list, dtype=torch.float32)

    # 기존 node_indicate 특성도 유지 (호환성을 위해)
    node_indicates = torch.ones(len(node_to_int_map), dtype=torch.float32).view(-1, 1)
    dgl_graph.ndata['node_indicate'] = node_indicates

    return dgl_graph





