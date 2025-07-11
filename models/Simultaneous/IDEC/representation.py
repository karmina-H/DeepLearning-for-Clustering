# -*- coding: utf-8 -*-
"""
This is a program for representation part of IDEC.
(Improved Deep Embedded Clustering with Local Structure Preservation)
Author: Guanbao Liang
License: BSD 2 clause
"""

from torch import nn


class StackedAutoEncoder(nn.Module):
    """
    This is a model that produces latent features from input features.
    (Improved Deep Embedded Clustering with Local Structure Preservation)

    Parameters
    ----------
    in_dim : int
        The feature dimension of the input data. -> input데이터의 차원크기
    dims : list[int] -> 스택형 오토인코더의 각 계층에 있는 유닛(뉴런)의 개수를 리스트 형태로 정의
        The numbers list of units in stacked autoencoder.
    """

    def __init__(self, in_dim, dims=None):
        super(StackedAutoEncoder, self).__init__()
        self.in_dim = in_dim
        self.dims = dims if dims else [500, 500, 2000, 10] # -> 마지막 잠재공간 크기는 10
        self.encoders = []
        self.decoders = []
        self.initialize()

    def initialize(self):
        """
        Function that initializes the model structure.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        n_input = self.in_dim
        for i, units in enumerate(self.dims, 1): # 인덱스 1부터 시작
            encoder_activation = nn.Identity() if i == len(self.dims) else nn.ReLU() # 마지막 잠재공간 거친후에는 활성화함수적용 안함.
            encoder = nn.Linear(n_input, units) # [n_input, units] 레이어 하나 만들고
            nn.init.normal_(encoder.weight, mean=0, std=0.01) # 초기 가중치를 평균이 0이고 표준편차가 0.01인 정규 분포에서 샘플링하여 초기화
            # 리스트에 layer하고 활성화 함수 차례대로 추가
            self.encoders.append(encoder) 
            self.encoders.append(encoder_activation)

            # decoder도 인코더와 그대로 똑같이
            decoder_activation = nn.Identity() if i == 1 else nn.ReLU()
            decoder = nn.Linear(units, n_input)
            nn.init.normal_(decoder.weight, mean=0, std=0.01)
            self.decoders.append(decoder_activation)
            self.decoders.append(decoder)

            #input크기는 units크기로 바꿔주고
            n_input = units
        # 리스트형태 encoders를 순서대로 연결해서 인코더네트워크로 만들어줌.
        self.encoder = nn.Sequential(*self.encoders)
        # 디코더는 순서뒤집어서 만들어줌
        self.decoders.reverse()
        self.decoder = nn.Sequential(*self.decoders)
        # 인코더-디코더 합쳐서 오토인코더 네트워크 만들어줌.
        self.autoencoder = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x):
        """
        Forward Propagation.

        Parameters
        ----------
        x : torch.Tensor
            The images.

        Returns
        -------
        encoded : torch.Tensor
            The encoded features.
        decoded : torch.Tensor
            The reconstructed input.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
