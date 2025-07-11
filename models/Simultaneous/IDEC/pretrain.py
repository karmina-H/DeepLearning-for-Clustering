# -*- coding: utf-8 -*-
"""
This is a program for pretrain part of IDEC.
(Improved Deep Embedded Clustering with Local Structure Preservation)
Author: Guanbao Liang
License: BSD 2 clause
"""

import math
import os
import sys

sys.path.append("./")
import time

import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from DataLoader import base_transforms
from representation import StackedAutoEncoder

from data.cv_dataset import load_torchvision_dataset
from utils.init_env import init, init_optimizer
from utils.log_save import save_param, save_train_details

from tqdm import tqdm


class PretrainSAE:
    """
    This is a model that pretrains Stacked-AutoEncoder.

    Parameters
    ----------
    in_dim : int
        The feature dimension of the input data.
    dims : list or int
        The list of numbers of units in stacked autoencoder.
    drop_fraction : float
        The rate of dropped features in layer-wise pretraining, defaults to 0.2.
    optimizer_name : str
        The optimizer specified by user.
    learning_rate : float
        The learning rate in the pretraining, defaults to 0.1.
    momentum : float
        The momentum of SGD optimizer, defaults to 0.9.
    weight_decay : float
        The value of weight decay of the model.
    batch_size : int
        The number of batch size in pretraining, defaults to 256.
    step_size : int
        The number of iterations that changes the learning rate.
    update_lr_rate : float
        The rate of changes the learning rate each step_size.
    n_iter_layer_wise : int
        The number of iteration in layer-wise pretraining, defaults to 50000.
    n_iter_fine_tuning : int
        The number of iteration in fine tuning, defaults to 100000.
    device_name : str
        The network will train on the device.
    log_dir : str
        The log directory.
    model_dir : str
        The saved model directory.
    save_step : int
        The model will be saved each save step.
    writer : SummaryWriter
        The tensorboard summary writer.
    verbose : bool
        Whether to print logs in console.
    """

    def __init__(
        self,
        in_dim,
        dims=None,
        drop_fraction=0.2,
        optimizer_name="sgd",
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=0.0,
        batch_size=512,
        step_size=20000,
        update_lr_rate=0.1,
        n_iter_layer_wise=50000,
        n_iter_fine_tuning=100000,
        device_name=None,
        log_dir=None,
        model_dir="",
        save_step=5000,
        writer=None,
        verbose=False,
    ):
        super(PretrainSAE, self).__init__()
        self.in_dim = in_dim
        self.dims = dims if dims else [500, 500, 2000, 10]
        self.drop_fraction = drop_fraction
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.update_lr_rate = update_lr_rate
        self.n_iter_layer_wise = n_iter_layer_wise
        self.n_iter_fine_tuning = n_iter_fine_tuning
        if device_name is None:
            device_name = "cpu"
        self.device = torch.device(device_name)
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.save_step = save_step
        self.writer = writer
        self.verbose = verbose
        self.initialize()

    def initialize(self):
        """
        Functions that initializes the stacked autoencoder model.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # 오토인코더 모델 초기화 후 gpu할당
        self.model = StackedAutoEncoder(self.in_dim, self.dims)
        self.model = self.model.to(self.device)

    def pretrain(self, X):
        """
        Functions that pretrains the stacked autoencoder model.

        Parameters
        ----------
        X : tensor
            The input data of the dataset.

        Returns
        -------
        None

        """
        n_iter_per_epoch = X.shape[0] // self.batch_size
        layer_wise_epochs = max(self.n_iter_layer_wise // n_iter_per_epoch, 1)
        fine_tuning_epochs = max(self.n_iter_fine_tuning // n_iter_per_epoch, 1)
        # 인코더가 레이어- 활성화함수 -레이어 -활성화함수로 구성되어있으니 pair크기는 나누기2해주어야함.
        num_pairs = len(self.model.encoders) // 2
        current_x = X

        self.model.decoders.reverse() # 밑에 왜 이거 사용하는지 나옴
        for i in range(num_pairs):
            # 인코더에서 레이어하나와 활성화 함수를 순서대로 하나씩 들고옴
            encoder, encoder_activation = (
                self.model.encoders[i * 2],
                self.model.encoders[i * 2 + 1],
            )
            # 디코더에서도 하나씩 들고옴
            decoder_activation, decoder = (
                self.model.decoders[i * 2],
                self.model.decoders[i * 2 + 1],
            ) 
            #들고온 것들로만 오토인코더 구성함, 드롭아웃도 추가해서 -> 이것때문에 위에 reverse사용함
            autoencoder = nn.Sequential(
                nn.Dropout(self.drop_fraction),
                encoder,
                encoder_activation,
                nn.Dropout(self.drop_fraction),
                decoder,
                decoder_activation,
            )
            #데이터 로드하고
            dataset = TensorDataset(current_x, current_x)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )
            # 평균제곱오차로 손실함수정의
            loss_func = nn.MSELoss()
            # 학습 옵티마이저 정의
            optimizer = init_optimizer(
                optimizer_name=self.optimizer_name,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                params=autoencoder.parameters(),
                sgd_momentum=self.momentum,
            )
            #스케쥴러도 정의함
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.step_size, gamma=self.update_lr_rate
            )

            pre_str = "Layer-wise/{} pair".format(i + 1)
            iteration_num = 1

            for epoch in range(1, layer_wise_epochs + 1):
                st_time = time.time()
                total_loss = 0
                total_sample_num = 0
                with tqdm(dataloader, desc=f"[Layer-wise] {i + 1} pair, Epoch {epoch}/{layer_wise_epochs}", unit="batch") as pbar:
                    for data, labels in pbar:
                        # train_one_batch -> 이거 오토인코더 레이어하나씩 학습하는 함수
                        # 로그용 loss값을 반환으로 받음"(역전파는 함수 안에서 실행됨)
                        loss = self.train_one_batch(
                            autoencoder,
                            data,
                            labels,
                            loss_func,
                            optimizer,
                            scheduler,
                            pre_str,
                            iteration_num,
                        )
                        total_loss = total_loss + loss * len(data)
                        total_sample_num = total_sample_num + len(data)
                        iteration_num = iteration_num + 1

                        pbar.set_postfix(batch_loss=f"{loss.item():.4f}")
                        
                avg_loss = total_loss / total_sample_num
                ed_time = time.time()
                train_details = "[Layer-wise] : {:2d} pair, [Epoch] : {:04d} / {:04d}, [Avg_loss] : {:.6f}, [time] : {:.3f} s ----- {:.2f}%".format(
                    i + 1,
                    epoch,
                    layer_wise_epochs,
                    avg_loss,
                    ed_time - st_time,
                    100 * (epoch) / layer_wise_epochs,
                )
                if self.verbose:
                    print(train_details)
                save_train_details(self.log_dir, train_details + "\n")

            # 하나의 학습끝난 레이어를 평가모드로 전환하고 detach로 이제 기울기 연산에서 제외함.
            # 그리고 다음 레이어 학습시 그전 레이어를 통과하되 그전레이어는 학습안되고 현재레이어만 학습되게 clone후 detach해줌
            # 학습이 완료된 현재 계층의 인코더만으로 구성된 임시 모델을 생성
            encoder_model = nn.Sequential(encoder) 
            encoder_model.eval()
            with torch.no_grad():
                # 이전 계층의 current_x를 현재 계층의 encoder에 통과시켜 다음 계층의 입력으로 사용할 current_x를 생성, detach사용해서 연산기록 끊어버림
                current_x = encoder_model(current_x.to(self.device)).clone().detach()
            # 해당 계층의 파라미터는 더 이상 이 단계에서 학습되지 않도록 requires_grad = False로 설정
            for param in autoencoder.parameters():
                param.requires_grad = False
        
        # 이러면 각 레이어가 각각 하나씩 학습 완료된것임
        # 그리고 전체 레이어를 한번에 다시학습 - 미세조정


        for param in self.model.autoencoder.parameters():
            param.requires_grad = True

        dataset = TensorDataset(X, X)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        loss_func = nn.MSELoss()
        optimizer = init_optimizer(
            optimizer_name=self.optimizer_name,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            params=self.model.autoencoder.parameters(),
            sgd_momentum=self.momentum,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.update_lr_rate
        )
        pre_str = "Fine-tuning"
        iteration_num = 1
        for epoch in range(1, fine_tuning_epochs + 1):
            st_time = time.time()
            total_loss = 0
            total_sample_num = 0
            for data, labels in dataloader:
                if iteration_num % self.save_step == 0:
                    ckpt = {
                        "net_model": self.model.autoencoder.state_dict(),
                        "sched": scheduler.state_dict(),
                        "optim": optimizer.state_dict(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "iteration_num": iteration_num,
                        "epoch": epoch,
                    }
                    if not os.path.exists(self.model_dir):
                        os.makedirs(self.model_dir)
                    torch.save(
                        ckpt,
                        os.path.join(self.model_dir, "ckpt_%d.pt" % (iteration_num)),
                    )

                loss = self.train_one_batch(
                    self.model.autoencoder,
                    data,
                    labels,
                    loss_func,
                    optimizer,
                    scheduler,
                    pre_str,
                    iteration_num,
                )
                total_loss = total_loss + loss * len(data)
                total_sample_num = total_sample_num + len(data)
                iteration_num = iteration_num + 1
            avg_loss = total_loss / total_sample_num
            ed_time = time.time()
            train_details = "[Fine-tuning], [Epoch] : {:04d} / {:04d}, [Avg_loss] : {:.6f}, [time] : {:.3f} s ----- {:.2f}%".format(
                epoch,
                fine_tuning_epochs,
                avg_loss,
                ed_time - st_time,
                100 * (epoch) / fine_tuning_epochs,
            )
            if self.verbose:
                print(train_details)
            save_train_details(self.log_dir, train_details + "\n")

    def train_one_batch(
        self,
        model,
        data,
        labels,
        loss_func,
        optimizer,
        scheduler=None,
        pre_str=None,
        iteration_num=None,
    ):
        """
        Functions that pretrains the stacked autoencoder model.

        Parameters
        ----------
        model : torch.nn.Module
            The network.
        data : tensor
            The input data.
        labels : tensor
            The ground-truth labels.
        loss_func : torch.nn.Module
            The loss function.
        optimizer : torch.optim.optimizer.Optimizer
            The optimizer.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            The scheduler.
        pre_str : str
            The string for the logger.
        iteration_num : int
            The number of the iteration.

        Returns
        -------
        loss : tensor
            The loss itemes.

        """
        model.train()
        data = data.to(self.device)
        labels = labels.to(self.device)

        outputs = model(data)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if pre_str is not None and iteration_num is not None:
            self.writer.add_scalar(pre_str, loss, iteration_num)

        return loss


args = init(config_file=["configs/base.yaml", "configs/IDEC.yaml"])
args.log_dir = f"{args.log_dir}/{args.dataset_name}/{args.method_name}/pretrain"
args.model_dir = f"{args.model_dir}/{args.dataset_name}/{args.method_name}/pretrain"
args.pretrain_path = f"{args.model_dir}/ckpt_100000.pt"
save_param(log_dir=args.log_dir, param_dict=vars(args), file_name="pretrain_params.txt")
writer = SummaryWriter(log_dir=args.log_dir)

trans = base_transforms(resize=args.img_size_at)
trainset = load_torchvision_dataset(
    dataset_name=args.dataset_name,
    dataset_dir=args.dataset_dir,
    train=True,
    transform=trans,
)

train_X = torch.cat([data for data, label in trainset], dim=0)
train_X = train_X.reshape(len(trainset), -1)
in_dim = train_X.shape[1]
Pretrainer = PretrainSAE(
    in_dim=in_dim,
    dims=[500, 500, 2000, 10],
    optimizer_name=args.optimizer,
    learning_rate=args.lr,
    momentum=args.sgd_momentum,
    batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    device_name=args.device,
    log_dir=args.log_dir,
    model_dir=args.model_dir,
    writer=writer,
    verbose=args.verbose,
    save_step=args.save_step,
)
Pretrainer.pretrain(train_X)
