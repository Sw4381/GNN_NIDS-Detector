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


class EnhancedGNNAutoencoder(nn.Module):
    def __init__(self, edge_dim, node_dim, hidden_dim=128, embedding_dim=64, latent_dim=16):
        """
        향상된 GNN 오토인코더 모델 - 이중 Aggregation 과정 포함 (단순화된 Aggregation)

        Args:
            edge_dim: 엣지 특성의 차원
            node_dim: 노드 특성의 차원
            hidden_dim: 공통된 임베딩 차원
            embedding_dim: 중간 임베딩 차원
            latent_dim: 최종 잠재 공간 차원
        """
        super(EnhancedGNNAutoencoder, self).__init__()
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        # 1. 노드와 엣지의 임베딩 차원을 동일하게 변환
        self.node_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        self.edge_projection = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        # 2. 첫 번째 집계된 메시지 처리를 위한 변환
        self.message_transform_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        # 3. 최종 엣지 임베딩 생성을 위한 변환
        self.edge_embedding = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),  # 엣지 + 집계된 노드 메시지
            nn.LeakyReLU(),
            nn.BatchNorm1d(embedding_dim)
        )

        # 4. 오토인코더 인코더 (3계층)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(embedding_dim // 2),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(embedding_dim // 4),
            nn.Linear(embedding_dim // 4, latent_dim)
        )

        # 5. 두 번째 집계된 메시지 처리를 위한 변환 (인코딩 이후)
        self.message_transform_2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(latent_dim)
        )

        # 6. 디코딩을 위한 최종 특성 통합
        self.final_fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),  # 인코딩된 엣지 + 두번째 집계 메시지
            nn.LeakyReLU(),
            nn.BatchNorm1d(latent_dim)
        )

        # 7. 오토인코더 디코더 (3계층)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, embedding_dim // 4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(embedding_dim // 4),
            nn.Linear(embedding_dim // 4, embedding_dim // 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(embedding_dim // 2),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Linear(embedding_dim, edge_dim),
            nn.Tanh()
        )

    def forward(self, graph, corrupt=False, augment=False, augmentation_level=0.3,
                aug_mode="hybrid", edge_drop_p=0.05, node_mask_p=0.03):
        """
        모델 순전파 - 단순화된 이중 Aggregation 과정 포함

        Args:
            graph: DGL 그래프
            corrupt: 부정 샘플 생성 여부
            augment: 데이터 증강 여부
            augmentation_level: 가우시안 노이즈 강도 (Sec. 3.1.3의 lambda)
            aug_mode: 증강 종류 — "gaussian" (feature noise only),
                      "edge_drop" (structural disconnection only),
                      "node_mask" (service-migration only),
                      "hybrid" (Gaussian + edge_drop + node_mask) [paper default]
            edge_drop_p: edge feature drop 확률 (Sec. 3.1.3의 p_e = 0.05)
            node_mask_p: node feature mask 확률 (Sec. 3.1.3의 p_n = 0.03)

        Returns:
            reconstructed_edge_features: 재구성된 엣지 특성
            latent_embeddings: 잠재 공간 임베딩 (오토인코더 은닉층)
        """
        # 작업용 그래프 복사
        working_graph = graph.clone()
        edge_features = working_graph.edata["feat"]
        node_features = working_graph.ndata["feat"]

        # 부정 샘플 생성 (corrupt=True인 경우)
        if corrupt:
            # 엣지와 노드 특성 모두 셔플링
            edge_idx = torch.randperm(edge_features.shape[0])
            node_idx = torch.randperm(node_features.shape[0])
            working_graph.edata["feat"] = edge_features[edge_idx]
            working_graph.ndata["feat"] = node_features[node_idx]

        # 하이브리드 데이터 증강 (augment=True인 경우) — Sec. 3.1.3
        if augment:
            use_gaussian = aug_mode in ("gaussian", "hybrid")
            use_edge_drop = aug_mode in ("edge_drop", "hybrid")
            use_node_mask = aug_mode in ("node_mask", "hybrid")

            # 1) Feature-level Gaussian noise (측정 잡음 모델링)
            if use_gaussian:
                edge_noise = torch.randn_like(edge_features) * augmentation_level * torch.std(edge_features, dim=0)
                working_graph.edata["feat"] = edge_features + edge_noise
                node_noise = torch.randn_like(node_features) * augmentation_level * torch.std(node_features, dim=0)
                working_graph.ndata["feat"] = node_features + node_noise

            # 2) Stochastic edge dropping (일시적 연결 실패 / 서비스 마이그레이션)
            #    곱셈 keep-mask로 적용 (in-place 회피 → autograd 버전 충돌 방지)
            if use_edge_drop and working_graph.num_edges() > 0:
                drop = (torch.rand(working_graph.num_edges(),
                                   device=working_graph.edata["feat"].device)
                        < edge_drop_p).float()
                keep = (1.0 - drop).unsqueeze(1)  # [E, 1]
                working_graph.edata["feat"] = working_graph.edata["feat"] * keep

            # 3) Node feature masking (샘플링/필터링 모니터링 하의 부분 가시성)
            if use_node_mask and working_graph.num_nodes() > 0:
                drop_n = (torch.rand(working_graph.num_nodes(),
                                     device=working_graph.ndata["feat"].device)
                          < node_mask_p).float()
                keep_n = (1.0 - drop_n).unsqueeze(1)  # [N, 1]
                working_graph.ndata["feat"] = working_graph.ndata["feat"] * keep_n

        with working_graph.local_scope():
            # 1. 노드와 엣지의 임베딩 차원을 동일하게 변환
            h_n = self.node_projection(working_graph.ndata["feat"])
            h_e = self.edge_projection(working_graph.edata["feat"])

            working_graph.ndata["h"] = h_n
            working_graph.edata["h"] = h_e

            # 2. 첫 번째 Aggregation - 단순화된 방식
            working_graph.edata["e_feat"] = h_e
            working_graph.update_all(fn.copy_e("e_feat", "msg"), fn.mean("msg", "h_v"))
            h_v = working_graph.ndata["h_v"]

            # 메시지 변환
            transformed_h_v = self.message_transform_1(h_v)
            working_graph.ndata["h_v"] = transformed_h_v

            # 3. 엣지 임베딩 계산
            src, dst = working_graph.edges()
            src_msgs = working_graph.ndata["h_v"][src]
            dst_msgs = working_graph.ndata["h_v"][dst]

            # 엣지의 양쪽 노드 메시지 평균 계산
            node_msgs = (src_msgs + dst_msgs) / 2

            # 엣지 특성과 집계된 노드 메시지 결합
            edge_inputs = torch.cat([h_e, node_msgs], dim=1)
            edge_embeddings = self.edge_embedding(edge_inputs)

            # 4. 오토인코더 인코더 적용
            latent_embeddings = self.encoder(edge_embeddings)

            # 5. 두 번째 Aggregation - 인코딩된 특성 기반, 단순화된 방식
            working_graph.edata["z_e"] = latent_embeddings
            working_graph.update_all(fn.copy_e("z_e", "msg"), fn.mean("msg", "h_z"))
            h_z = working_graph.ndata["h_z"]

            # 인코딩된 메시지 변환
            transformed_h_z = self.message_transform_2(h_z)
            working_graph.ndata["h_z"] = transformed_h_z

            # 6. 두 번째 엣지 임베딩 계산
            src_msgs_2 = working_graph.ndata["h_z"][src]
            dst_msgs_2 = working_graph.ndata["h_z"][dst]

            # 두 번째 엣지의 양쪽 노드 메시지 평균 계산
            node_msgs_2 = (src_msgs_2 + dst_msgs_2) / 2

            # 인코딩된 엣지 특성과 두 번째 집계된 노드 메시지 결합
            final_inputs = torch.cat([latent_embeddings, node_msgs_2], dim=1)
            final_latent = self.final_fusion(final_inputs)

            # 7. 디코더 적용하여 재구성
            reconstructed_edge_embeddings = self.decoder(final_latent)

        return reconstructed_edge_embeddings, latent_embeddings




class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features




class EnhancedGNNModel(nn.Module):
    def __init__(self, edge_dim, node_dim, hidden_dim=128, embedding_dim=64, latent_dim=16,
                 num_augmentations=3, direction="both", lambda_dgi=1.0,
                 aug_mode="hybrid", edge_drop_p=0.05, node_mask_p=0.03,
                 augmentation_levels=(0.2, 0.4, 0.6)):
        """
        향상된 GNN 모델 (단순화된 이중 Aggregation 과정 포함) - 포커스 학습 제거

        Args:
            edge_dim: 엣지 특성의 차원
            node_dim: 노드 특성의 차원
            hidden_dim: 공통된 임베딩 차원
            embedding_dim: 중간 임베딩 차원
            latent_dim: 최종 잠재 공간 차원
            num_augmentations: 증강 데이터 개수 (Sec. 3.1.3의 N)
            direction: 메시지 전달 방향 (무시됨 - 단순화를 위해)
            lambda_dgi: DGI loss 가중치
            aug_mode: "gaussian" | "edge_drop" | "node_mask" | "hybrid" [paper default]
            edge_drop_p: edge feature drop 확률 (Sec. 3.1.3의 p_e = 0.05)
            node_mask_p: node feature mask 확률 (Sec. 3.1.3의 p_n = 0.03)
            augmentation_levels: 가우시안 노이즈 강도 스케줄 (Sec. 3.1.3의 Lambda = {0.2, 0.4, 0.6})
        """
        super(EnhancedGNNModel, self).__init__()
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.lambda_dgi = lambda_dgi

        # 증강 긍정 샘플 개수
        self.num_augmentations = num_augmentations

        # 메시지 전달 방향 (사용하지 않음)
        self.direction = direction

        # 하이브리드 증강 설정 (Sec. 3.1.3)
        self.aug_mode = aug_mode
        self.edge_drop_p = edge_drop_p
        self.node_mask_p = node_mask_p
        self.augmentation_levels = tuple(augmentation_levels)

        # 향상된 GNN Autoencoder 컴포넌트
        self.autoencoder = EnhancedGNNAutoencoder(
            edge_dim=edge_dim, 
            node_dim=node_dim, 
            hidden_dim=hidden_dim, 
            embedding_dim=embedding_dim, 
            latent_dim=latent_dim
        )

        # DGI Discriminator 컴포넌트
        self.discriminator = Discriminator(latent_dim)

        # Loss functions
        self.reconstruction_loss_fn = nn.MSELoss(reduction='none')
        self.dgi_loss = nn.BCEWithLogitsLoss()

    def forward(self, graph, include_augmented_recon=True, training=True):
        """
        Forward pass with option to include augmented reconstruction loss

        Args:
            graph: Input graph
            include_augmented_recon: Whether to include augmented data reconstruction loss
            training: Whether the model is in training mode
        """
        # 원본 긍정 샘플 처리
        reconstructed_edge_attr, z_uv_positive_original = self.autoencoder(
            graph, 
            corrupt=False,
            augment=False
        )

        # 원본 데이터에 대한 재구성 손실 계산
        edge_features = graph.edata["feat"]
        per_edge_errors = self.reconstruction_loss_fn(reconstructed_edge_attr, edge_features)
        per_edge_mse = per_edge_errors.mean(dim=1)
        recon_loss = per_edge_mse.mean()

        # 여러 증강된 긍정 샘플 생성 (Sec. 3.1.3 하이브리드 증강)
        augmented_embeddings = []
        augmented_reconstructions = []
        augmented_edge_features = []  # 증강된 엣지 특성 저장
        augmentation_recon_losses = []
        augmentation_levels = list(self.augmentation_levels)  # 가우시안 노이즈 강도 스케줄

        if training:  # 학습 모드에서만 데이터 증강 수행
            for i in range(self.num_augmentations):
                # 매 반복마다 다른 증강 강도 사용
                aug_level = augmentation_levels[i % len(augmentation_levels)]

                # 증강 그래프 생성 (autoencoder내에서 하이브리드 증강 수행)
                aug_reconstructed, z_uv_augmented = self.autoencoder(
                    graph,
                    corrupt=False,
                    augment=True,
                    augmentation_level=aug_level,
                    aug_mode=self.aug_mode,
                    edge_drop_p=self.edge_drop_p,
                    node_mask_p=self.node_mask_p,
                )

                # 증강된 엣지 특성 (loss target).
                # Gaussian 모드: noisy 타겟. 그 외(edge_drop/node_mask/hybrid): 원본 타겟
                # (denoising paradigm — perturbed input에서 원본 복원).
                if self.aug_mode == "gaussian":
                    noise = torch.randn_like(edge_features) * aug_level * torch.std(edge_features, dim=0)
                    aug_edge_features = edge_features + noise
                else:
                    aug_edge_features = edge_features

                augmented_embeddings.append(z_uv_augmented)
                augmented_reconstructions.append(aug_reconstructed)
                augmented_edge_features.append(aug_edge_features)

                # 증강된 데이터에 대한 재구성 손실 계산 (증강된 엣지 특성과 비교)
                if include_augmented_recon:
                    aug_per_edge_errors = self.reconstruction_loss_fn(aug_reconstructed, aug_edge_features)
                    aug_per_edge_mse = aug_per_edge_errors.mean(dim=1)
                    aug_recon_loss = aug_per_edge_mse.mean()
                    augmentation_recon_losses.append(aug_recon_loss)

        # 부정 샘플 처리 (랜덤 섞기만 사용) - 학습 모드일 때만
        if training:
            # corrupt=True로 설정하여 부정 샘플 생성
            _, z_uv_negative = self.autoencoder(
                graph, 
                corrupt=True, 
                augment=False
            )
        else:
            z_uv_negative = None

        # 증강된 데이터의 재구성 손실을 총 재구성 손실에 포함
        if include_augmented_recon and augmentation_recon_losses and training:
            # 원본 + 증강 데이터의 평균 재구성 손실
            total_recon_loss = (recon_loss + sum(augmentation_recon_losses)) / (1 + len(augmentation_recon_losses))
        else:
            # 원본 데이터만의 재구성 손실
            total_recon_loss = recon_loss

        # 검증 모드일 때는 DGI 손실을 계산하지 않고 재구성 손실만 반환
        if not training:
            return {
                'total_loss': total_recon_loss,
                'reconstruction_loss': recon_loss,
                'dgi_loss': 0.0,
                'embeddings': z_uv_positive_original,
                'reconstructed_features': reconstructed_edge_attr,
                'per_edge_mse': per_edge_mse
            }

        # --- 여기서부터는 학습 모드일 때만 실행 ---

        # 원본 임베딩에 대한 그래프 요약 생성
        summary_original = torch.sigmoid(z_uv_positive_original.mean(dim=0))

        # 증강된 임베딩에 대한 그래프 요약 생성
        if augmented_embeddings:
            augmented_tensor = torch.cat(augmented_embeddings, dim=0)
            summary_augmented = torch.sigmoid(augmented_tensor.mean(dim=0))

        # 원본 샘플과 부정 샘플 간의 DGI 손실
        positive_original = self.discriminator(z_uv_positive_original, summary_original)
        negative_vs_original = self.discriminator(z_uv_negative, summary_original)

        dgi_loss_original = (
            self.dgi_loss(positive_original, torch.ones_like(positive_original)) + 
            self.dgi_loss(negative_vs_original, torch.zeros_like(negative_vs_original))
        )

        # 증강된 샘플과 부정 샘플 간의 DGI 손실
        dgi_loss_augmented = 0
        if augmented_embeddings:
            for z_uv_aug in augmented_embeddings:
                positive_aug = self.discriminator(z_uv_aug, summary_augmented)
                dgi_loss_augmented += self.dgi_loss(positive_aug, torch.ones_like(positive_aug))

            negative_vs_augmented = self.discriminator(z_uv_negative, summary_augmented)
            dgi_loss_augmented += self.dgi_loss(negative_vs_augmented, torch.zeros_like(negative_vs_augmented))

            # 증강된 샘플 개수 + 1(부정 샘플)로 나누어 평균 계산
            dgi_loss_augmented /= (len(augmented_embeddings) + 1)

        # 총 DGI 손실 (원본과 증강된 샘플에 대한 손실의 평균)
        if augmented_embeddings:
            dgi_loss = (dgi_loss_original + dgi_loss_augmented) / 2
        else:
            dgi_loss = dgi_loss_original

        # 전체 손실
        total_loss = total_recon_loss + self.lambda_dgi * dgi_loss

        return {
            'total_loss': total_loss,
            'reconstruction_loss': total_recon_loss,
            'original_recon_loss': recon_loss,
            'augmented_recon_losses': augmentation_recon_losses if augmentation_recon_losses else [0],
            'dgi_loss': dgi_loss,
            'dgi_loss_original': dgi_loss_original,
            'dgi_loss_augmented': dgi_loss_augmented if augmented_embeddings else 0,
            'embeddings': z_uv_positive_original,
            'reconstructed_features': reconstructed_edge_attr,
            'per_edge_mse': per_edge_mse,
            'augmented_embeddings': augmented_embeddings,
            'augmented_reconstructions': augmented_reconstructions,
            'augmented_edge_features': augmented_edge_features
        }

    def training_step(self, graph, optimizer):
        """
        일반적인 학습 단계 - 포커스 학습 없음

        Args:
            graph: 학습할 그래프
            optimizer: 옵티마이저

        Returns:
            loss_info: 손실 관련 정보 딕셔너리
        """
        # 학습 모드에서 증강된 데이터에 대한 재구성 손실 포함
        output = self(graph, include_augmented_recon=True, training=True)

        total_loss = output['total_loss']

        # 옵티마이저 단계 수행
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': output['reconstruction_loss'].item(),
            'original_recon_loss': output['original_recon_loss'].item() if 'original_recon_loss' in output else output['reconstruction_loss'].item(),
            'dgi_loss': output['dgi_loss'].item()
        }
