# -*- coding: utf-8 -*-
"""
This is a program for clustering and loss part of IDEC.
(Improved Deep Embedded Clustering with Local Structure Preservation)
Author: Guanbao Liang
License: BSD 2 clause
"""

import torch

from torch import nn


# 입력으로 들어오는 잠재 특징(latent features)이 각 클러스터에 속할 확률을 계산함. 이 확률 계산은 Student's t-분포의 변형된 형태를 사용
class Clustering(nn.Module):
    """
    This is a model that calculates the probability of the sample belonging to each cluster.
    (Improved Deep Embedded Clustering with Local Structure Preservation)

    Parameters
    ----------
    in_dim : int
        The feature dimension of the input data.
    n_clusters : int
        The number of clusters.
    alpha : float
        The parameter in Student's t-distribution which defaults to 1.0.
    weights : numpy.ndarray
        The weights of centroids which is obtained by Kmeans.

    Examples
    --------
    # >>> model = Clustering(in_dim=784,n_clusters=10,alpha=1.0,weights=None)
    # >>> out = model(input_data)
    """

    def __init__(self, in_dim, n_clusters, alpha=1.0, weights=None):
        super(Clustering, self).__init__()
        self.n_clusters = n_clusters # 클러스터(군집)의 개수
        self.alpha = alpha # Student's t-분포의 매개변수 (자유도와 관련)
        self.centroids = nn.Parameter( # 각 클러스터의 중심(Centroid)을 나타내는 학습 가능한 파라미터
            torch.empty(n_clusters, in_dim), requires_grad=True
        )
        self.initial_weights = weights  # K-평균 등으로 초기화된 중심 값
        self.initialize() # 중심 초기화 함수 호출

    def initialize(self):
        """
        Functions that initializes the centroids.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # 기본적으로 Xavier 균등 분포로 중심 초기화
        nn.init.xavier_uniform_(self.centroids)
        # 외부에서 초기 중심(weights)이 주어졌다면 해당 값으로 덮어씀
        if self.initial_weights is not None:
            weights_tensor = torch.tensor(self.initial_weights).float()
            self.centroids.data = weights_tensor

    def forward(self, inputs):
        """
        Function that calculates the probability of assigning sample i to cluster j.

        Parameters
        ----------
        inputs : torch.Tensor
            The data you input.

        Returns
        -------
        q : torch.Tensor
            The data of probabilities of assigning all samples to all clusters.
        """
        # inputs = 오토인코더를 통과한 잠재벡터
        # Student's t-분포 기반 확률 계산 ->즉  오토인코더의 인코딩된 잠재 특징이 각 클러스터 중심에 얼마나 속하는지 나타내는 확률을 계산하는 것.
        # input = (batch_size, in_dim)
        q = 1.0 / (
            1.0
            + (
                torch.sum( # (batch_size, 1, in_dim)이거로 바꾸는 이유가 centroids는 (n_clusters, in_dim) 이건데 (B, 1, D) - (N, D)이렇게 차원맞춰주면 centriods를 (1, N, D)이렇게 브로드개스팅해서
                           #  (batch_size, n_clusters, in_dim)이 형태의 결과를 얻을 수 있음
                           # 그리고 in_dim차원에 대해서 값을 sum 즉 i번째 배치에서 j번째 클러스터링 중심사이거리제곱의 합
                           # (batch_size, n_clusters)으로 결과반환
                    torch.square(torch.unsqueeze(inputs, dim=1) - self.centroids), dim=2
                )
                / self.alpha
            )
        )
        q = q ** ((self.alpha + 1.0) / 2.0)
        # torch.sum(q, dim=1) ->  클러스터 차원)에 대해 q의 모든 값을 합산 =  (batch_size,)
        # 그리고 (n_clusters, batch_size)이게 분자고 (batch_size,) 이게 분모임 transpose(q, 0, 1) 이거 해주는 이유가 브로드캐스팅 올바르게 되려면
        # (n_clusters, batch_size)로 만들면, (batch_size,) 형태의 합계 텐서가 (1, batch_size)로 브로드캐스팅되니까
        # 샘플  즉 배치 i에 대해, 각 클러스터 j에 대한 유사성 $q_{ij}$를 해당 샘플의 모든 클러스터에 대한 유사성 총합으로 나눕니다. 즉, 각 샘플에 대한 클러스터 할당 확률 분포를 만듭
        q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, dim=1), 0, 1)
        return q


class Loss(nn.Module):
    """
    This is a Loss object.

    Parameters
    ----------
    gamma : float
        The coefficient of cluster loss.
    """

    def __init__(self, gamma=1.0):
        super(Loss, self).__init__()
        self.recon_loss = nn.MSELoss()
        self.cluster_loss = nn.KLDivLoss(reduction="batchmean")
        self.gamma = gamma

    def forward(self, pred, target, recon, x):
        """
        Loss calculation.

        Parameters
        ----------
        pred : torch.Tensor
            The model predictions after logarithm.
        target : torch.Tensor
            The ground-truth labels.
        recon : torch.Tensor
            The reconstructed input.
        x : torch.Tensor
            The original input.

        Returns
        -------
        losses : torch.Tensor
            The losses calculated by loss function.
        """
        lossC = self.cluster_loss(pred, target) # kl다이버전스를 이용한 클러스터링 손실
        lossR = self.recon_loss(recon, x) * self.gamma # 재구성 손실
        return lossC + lossR
