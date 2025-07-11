# -*- coding: utf-8 -*-
"""
This is a program for reproducing IDEC.
(Improved Deep Embedded Clustering with Local Structure Preservation)
Author: Guanbao Liang
License: BSD 2 clause
"""

import datetime
import sys

sys.path.append("./")
import time

import numpy as np
import torch

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans

from DataLoader import base_transforms
from representation import StackedAutoEncoder
from clustering import Clustering, Loss
from data.cv_dataset import load_torchvision_dataset
from utils.init_env import init, init_optimizer, init_backbone
from utils.evaluation import eva
from utils.log_save import (
    save_param,
    save_train_details,
    save_eva_details,
    save_model,
    save_json,
)


class IDEC(nn.Module):
    """
    This is the entire IDEC components.
    (Improved Deep Embedded Clustering with Local Structure Preservation)


    Parameters
    ----------
    in_dim : int
        The feature dimension of the input data.
    n_clusters : int
        The number of clusters.
    n_init : int
        The number of kmeans executions.
    max_iter : int
        The max iteration of Kmeans algorithm.
    tol : float
        The tolerance of Kmeans algorithm.
    alpha : float
        The parameter in Student's t-distribution which defaults to 1.0.
    pretrain : bool
        Whether to choice the pretrain mode.
    pretrain_path : str
        The place of the pretrain model.
    device_name : str
        The network will train on the device.
    """

    def __init__(
        self,
        in_dim=784, #28*28크기의 이미지를 사용해서
        n_clusters=10, # 이거 기본이 10인데 바꾸려면 오토인코더의 잠재공간의 크기도 바꿔주어야함
        n_init=20,
        max_iter=20000,
        tol=0.001,
        alpha=1.0,
        pretrain=True,
        pretrain_path=None,
        device_name="cuda",
        reuse="True",
    ):
        super(IDEC, self).__init__()
        self.in_dim = in_dim
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.pretrain = pretrain
        self.pretrain_path = pretrain_path
        self.device = device_name
        self.reuse = reuse

    def init_pretrain(self, x):
        """
        Funtion that initializes the stacked autoencoder and clustering layer.
        (스택형 오토인코더와 클러스터링 레이어를 초기화하는 함수)
        Parameters
        ----------
        x : torch.Tensor
            The images.

        Returns
        -------
        None
        """
        # pretrain된 오토인코더 파라미터 로드
        state_dict = torch.load(self.pretrain_path)
        representation = StackedAutoEncoder(self.in_dim)
        autoencoder = representation.autoencoder
        if self.reuse:
            autoencoder.load_state_dict(state_dict["net_model"])
        # 인코더부분만 불러옴(디코더뺴고)
        self.representaion = autoencoder[0].to(self.device)
        
        # 평가모드로 전환하고
        self.representaion.eval()
        # 입력데이터 x를 인코더에 넣어서 잠재벡터로 변환 -> 이거 기울기연산 x
        with torch.no_grad():
            x = self.representaion(x.to(self.device)).cpu()

        #k-means객체 생성하고
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        # K-평균을 잠재 특징 x에 피팅하고, 각 샘플의 클러스터 할당(labels)을 예측
        self.labels = torch.tensor(
            kmeans.fit_predict(x.cpu().numpy()), dtype=torch.float, requires_grad=True
        )
        # # K-평균으로 계산된 클러스터 중심을 가져옴
        cluster_centers = kmeans.cluster_centers_
        # 오토인코더에서 디코더까지 전부가져오고
        self.representaion = representation.to(self.device)

        # 클러스터링 인스턴스 초기화
        # # `x.shape[1]`은 잠재 특징의 차원 (즉, clustering 레이어의 in_dim), 여기서는 10으로 되어있음
        self.clustering = Clustering(
            x.shape[1], self.n_clusters, self.alpha, cluster_centers
        ).to(self.device)
        self.built = True

    def forward(self, x):
        """
        Forward Propagation.

        Parameters
        ----------
        x : torch.Tensor
            The images.

        Returns
        -------
        c : torch.Tensor
            Clustering assignments.
        recon_x : torch.Tensor
            The reconstructed input.
        """
        # 오토인코더에 통과시키고
        feat_encode, recon_x = self.representaion(x)
        # 잠재벡터를 clustering에 통과시키고
        c = self.clustering(feat_encode)
        # 클러스터할당확률인 Q와 재구성된 x반환
        return c, recon_x

    def predict(self, x):
        """
        Function that calculates the probability of assigning sample i to cluster j

        Parameters
        ----------
        x : torch.Tensor
            The node features matrix.

        Returns
        -------
        out : torch.Tensor
            The clustering assignment.
        """
        with torch.no_grad():
            feat_encode, recon_x = self.representaion(x)
            c = self.clustering(feat_encode)
            return torch.argmax(c, dim=1)


args = init(config_file=["configs/base.yaml", "configs/IDEC.yaml"])
args.log_dir = f"{args.log_dir}/{args.dataset_name}/{args.method_name}/finetuing"
args.model_dir = f"{args.model_dir}/{args.dataset_name}/{args.method_name}/finetuing"
save_param(log_dir=args.log_dir, param_dict=vars(args))
writer = SummaryWriter(log_dir=args.log_dir)

trans = base_transforms(resize=args.img_size_at)
dataset = load_torchvision_dataset(
    args.dataset_name,
    args.dataset_dir,
    train=True,
    transform=trans,
)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=False,
)

dataset_eval = load_torchvision_dataset(
    args.dataset_name,
    args.dataset_dir,
    train=True,
    transform=trans,
)
dataloader_eval = DataLoader(
    dataset=dataset_eval,
    batch_size=args.eval_batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    drop_last=False,
)

train_X = torch.cat([data for data, label in dataset], dim=0)
train_X = train_X.reshape(len(dataset), -1) # 데이터를 1차원 벡터로 평탄화 (28x28 이미지 -> 784 벡터)
in_dim = train_X.shape[1]

criterion = Loss().to(args.device) # IDEC Loss 함수 (재구성 손실 + 클러스터링 손실) 초기화
model = IDEC(
    in_dim=in_dim,
    n_clusters=args.class_num,
    pretrain_path=args.pretrain_path,
    device_name=args.device,
).to(args.device)

model.init_pretrain(train_X)
optimizer = init_optimizer(
    optimizer_name=args.optimizer,
    lr=args.lr,
    weight_decay=args.weight_decay,
    params=model.parameters(),
    sgd_momentum=args.sgd_momentum,
)
best_acc = -1
important_info = dict()
important_info["dataset"] = args.dataset_name
important_info["class_num"] = args.class_num
important_info["model"] = args.method_name
important_info["optimizer"] = args.optimizer
important_info["epoch"] = -1
important_info["acc"] = best_acc
important_info["traning_type"] = "finetuing"
for epoch in range(args.start_epoch, args.epochs + 1):
    model.train()
    total_examples = 0
    total_loss = 0
    start_time = time.time()
    for step, (data, _) in enumerate(dataloader):
        small_start_time = time.time() #학습시간 측정시작
        data = data.to(args.device) # 데이터를 gpu로 이동
        data = data.reshape(data.shape[0], -1) # 데이터 평탄화 : 28*28 -> 784
        # forward
        q, recon_x = model(data)
        #  # 목표 분포 P 계산 (DEC 논문의 P 계산 방식)
        with torch.no_grad():
            # `q.sum(dim=0)`: 각 클러스터에 할당된 모든 샘플의 q 값 합계 (크기: n_clusters)
            # q^2: 각 q 값을 제곱 (크기: batch_size, n_clusters)
            # weight: q^2를 각 클러스터의 q 값 합계로 나눔 (크기: batch_size, n_clusters)
            # 이는 각 클러스터가 샘플에 미치는 영향력을 강조하는 가중치입니다.
            weight = q**2 / q.sum(dim=0)
            # weight.T: 전치하여 (n_clusters, batch_size)
            # weight.sum(dim=1): 각 샘플에 대해 모든 클러스터에 대한 weight의 합 (크기: batch_size)
            # (weight.T / ...): 각 샘플의 weight를 그 샘플의 총 weight 합으로 정규화
            # .T: 다시 전치하여 (batch_size, n_clusters) 형태로 복구
            # 결과 p는 각 샘플에 대한 "목표 확률 분포"입니다. 이 분포는 `q`보다 더 '단단하고' '확신하는' 형태를 띕니다.
            # 클러스터링 학습은 q를 p에 가깝게 만드는 방향으로 진행됩니다.
            p = (weight.T / weight.sum(dim=1)).T
        # 손실계산
        loss = criterion(q.log(), p, recon_x, data)
        total_loss += loss.item() * data.size(0)
        total_examples += data.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        small_end_time = time.time()
        train_details = "[Epoch : {:04d}/{:04d}] ,[Mini-batch : {:04d}/{:04d}], [total_loss : {:.7f}], [time : {:.3f}s]".format(
            epoch - args.start_epoch + 1,
            args.epochs - args.start_epoch + 1,
            step + 1,
            len(dataloader),
            loss.item(),
            small_end_time - small_start_time,
        )
        if args.verbose:
            print(train_details)
        save_train_details(args.log_dir, train_details + "\n")
    avg_loss = total_loss / total_examples
    writer.add_scalar("Loss/train", avg_loss, epoch)
    end_time = time.time()
    train_details = "[Epoch : {:04d}/{:04d}], [loss : {:.7f}], [time : {:.3f}s]".format(
        epoch, args.epochs, avg_loss, end_time - start_time
    )
    if args.verbose:
        print(train_details)
    save_train_details(args.log_dir, train_details + "\n\n")

    if epoch % args.eval_step == 0:
        if args.verbose:
            print("\nEvaluate the model is starting...")
        model.eval()
        feature_vector = []
        labels_vector = []
        with torch.no_grad():
            for step, (x, y) in enumerate(dataloader_eval):
                x = x.to(args.device)
                x = x.view(x.size(0), -1)
                c = model.predict(x)
                feature_vector.extend(c.cpu().numpy())
                labels_vector.extend(y.cpu().numpy())

        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector)
        acc, f1, nmi, ari = eva(labels_vector, feature_vector)

        writer.add_scalar("Eval/acc", acc, epoch)
        writer.add_scalar("Eval/f1", f1, epoch)
        writer.add_scalar("Eval/nmi", nmi, epoch)
        writer.add_scalar("Eval/ari", ari, epoch)
        eval_details = "[Epoch : {:04d}/{:04d}], [acc : {:.4f}%], [f1 : {:.4f}%], [nmi : {:.4f}%], [ari : {:.4f}%]".format(
            epoch,
            args.epochs - args.start_epoch + 1,
            acc * 100,
            f1 * 100,
            nmi * 100,
            ari * 100,
        )
        save_eva_details(args.log_dir, eval_details + "\n\n")
        if args.verbose:
            print("Evaluate the model is over...\n")
            print(eval_details)
        if best_acc < acc:
            best_acc = acc
            important_info["acc"] = best_acc
            important_info["f1"] = f1
            important_info["nmi"] = nmi
            important_info["ari"] = ari
            important_info["epoch"] = epoch
            save_json(args.log_dir, important_info)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_size = 0
            for param in model.parameters():
                if param.requires_grad:
                    model_size += param.data.nelement()
            ckpt = {
                "current_time": current_time,
                "args": vars(args),
                "iteration_num": epoch,
                "model_size": "{:.2f} MB".format(model_size / 1024 / 1024),
                "net_model": model.state_dict(),
                "lr": optimizer.param_groups[0]["lr"],
                "optimizer": optimizer.state_dict(),
            }
            save_model(args.model_dir, ckpt, epoch)

    if epoch % args.save_step == 0:
        if args.verbose:
            print("\nSave the model is starting...")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_size = 0
        for param in model.parameters():
            if param.requires_grad:
                model_size += param.data.nelement()
        ckpt = {
            "current_time": current_time,
            "args": vars(args),
            "iteration_num": epoch,
            "model_size": "{:.2f} MB".format(model_size / 1024 / 1024),
            "net_model": model.state_dict(),
            "lr": optimizer.param_groups[0]["lr"],
            "optimizer": optimizer.state_dict(),
        }
        save_model(args.model_dir, ckpt, epoch)
        if args.verbose:
            print("Save the model is over...\n")
