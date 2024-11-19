# user zhengkelong

from modified_model.models import Model
from modified_model.utils.data import MyDataset, Data

from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import torch.optim as optim
import time
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional, List
from torch.utils.tensorboard import SummaryWriter
import math
from modified_model.utils.helper import get_optimizer, get_scheduler

from safetensors import safe_open
from safetensors.torch import save_file
from modified_model.utils.eRank import train_eRank


fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s-%(lineno)s"
logging.basicConfig(format=fmt)
run_logger = logging.getLogger(__name__)


@dataclass
class DataArgs:
    data_file: str = "/media/zkl/zkl_T7/preprocession/observe_prediction_512"
    data_save_path: str = "./modified_model/datas"


@dataclass
class TrainArgs:
    model_save_path: str = "./modified_model/checkpoints"
    save_epoch_freq: int = 1
    epochs: int = 100
    start_epoch: int = 0
    batch_size: int = 32
    lr: float = 1e-2
    lrs: List[float] = field(default_factory=list)

    loss_iter_log: int = 2
    plot_test_fig: bool = True

    seed: int = 123
    warm_up: bool = True
    warmup_steps: int = 20
    lr_min: float = 1e-8
    scheduler: str = "cosine"
    optimizer: str = "adamw"

    device: str = "cuda"
    ratio: float = 0.7
    resume: bool = True
    saving_fig_log_freq: int = 10

    using_eRank: bool = False
    eRank_lr: float = 1e-3
    eRank_epochs: int = 100


# 采用 typing 中的，必须添加 List[int]
@dataclass
class ModelArgs:
    e_layer: int = 3
    e_layers: List[int] = field(default_factory=list)
    pred_len: int = 512
    output_attention: bool = False
    enc_in: int = 1
    d_models: List[int] = field(default_factory=list)
    d_model: int = 1024
    embed: str = "fixed"
    freq: str = "h"
    dropout: float = 0.1
    d_ff: Optional[int] = None
    activation: str = "glue"
    exp_setting: int = 2
    c_out: int = 1
    n_heads: int = 5
    factor: int = 3


# @dataclass
# class ModelArgs():
#     e_layer: int = 3
#     e_layers: List[int] = field(default_factory=lambda: [])
#     pred_len: int = 512
#     output_attention: bool = False
#     enc_in: int = 1
#     d_models: List[int] = field(default_factory=lambda: [])
#     d_model: int = 1024
#     embed: str = 'fixed'
#     freq: str = 'h'
#     dropout: float = 0.1
#     d_ff: Optional[int] = None
#     activation: str = 'glue'
#     exp_setting: int = 2
#     c_out: int = 1
#     n_heads: int = 5
#     factor: int = 3


def run(train_args, data_args, model_args, dtype=torch.float, loss=torch.nn.MSELoss()):
    n_layers = model_args.e_layers
    batch_size = train_args.batch_size
    epochs = train_args.epochs
    start_epoch = train_args.start_epoch
    # list
    lrs = train_args.lrs

    # E:\model

    device = torch.device(train_args.device)
    model_save_file = train_args.model_save_path
    data_file = data_args.data_file

    ratio = train_args.ratio
    seed = train_args.seed

    if not os.path.exists(data_args.data_save_path):
        os.mkdir(data_args.data_save_path)
    data_save_file = data_args.data_save_path

    tempt = os.path.join(os.path.join(data_save_file), "mapping.json")
    if os.path.exists(tempt):
        index_mapping_file = tempt
        load_from_file = os.path.join(data_save_file, "train_val_test.npz")
    else:
        load_from_file = None
        index_mapping_file = None

    tempt = os.path.join(model_save_file, "model_params")
    if os.path.exists(tempt):
        model_save_file = tempt
    else:
        os.mkdir(tempt)
        model_save_file = tempt

    tempt = os.path.join(model_save_file, "summary_logs")
    if not os.path.exists(tempt):
        os.mkdir(tempt)
        summary_dir = tempt
    else:
        summary_dir = tempt

    tempt = os.path.join(model_save_file, "train_loss_history")
    if not os.path.exists(tempt):
        os.mkdir(tempt)
        train_loss_history_path = tempt
    else:
        train_loss_history_path = tempt

    writer = SummaryWriter(log_dir=summary_dir, flush_secs=15)

    dataset = Data(
        file_path=data_file,
        loading_from_file=load_from_file,
        index_mapping_file=index_mapping_file,
        ratio=ratio,
        seed=seed,
    )
    # input,label,depth
    # batch_size=dataset.train_len

    train_set_loader, val_set_loader, test_set_loader = dataset.get_dataloaders(
        batch_size=batch_size, dtype=dtype
    )
    _, _, test_set_loader_for_save_fig = dataset.get_dataloaders(
        batch_size=1, dtype=dtype
    )
    print(f"length of val_set_dataloader:{len(val_set_loader)}")
    n_layers = model_args.e_layers
    for n_layer in n_layers:
        model_args.e_layer = n_layer
        for d_model in model_args.d_models:
            model_args.d_model = d_model
            model = Model(configs=model_args).to(device=device)
            # writer.add_graph(model)
            for lr in train_args.lrs:
                train_args.lr = lr
                worse_model = Model(configs=model_args).to(device=device)
                for param in worse_model.parameters():
                    param.requires_grad = False

                if train_args.using_eRank:
                    run_logger.info(f"using eRank training")
                    params_group = []
                    for name, param in model.named_parameters():
                        if (
                            "projection_head.weight" in name
                            or "projection_head.bias" in name
                        ):
                            param.requires_grad = False
                            run_logger.debug(f"the name of param:{name}")
                        else:
                            params_group.append(param)
                    model = model.to(device=device)
                    eRank_optimizer = optim.AdamW(params_group, lr=train_args.eRank_lr)
                    run_logger.info(f"using eRank optimizer")
                    kwargs = {"e_layer": n_layer, "lr": lr, "d_model": d_model}
                    train_eRank(
                        better_model=model,
                        worse_model=worse_model,
                        train_loader=train_set_loader,
                        optimizer=eRank_optimizer,
                        train_args=train_args,
                        writer=writer,
                        **kwargs,
                    )
                    run_logger.info(f"finish eRank training++++++++++++++++++++++++++")
                    run_logger.info(f"resume training the model")
                    for param in model.parameters():
                        param.requires_grad = True

                optimizer = get_optimizer(model=model, train_args=train_args)
                if start_epoch > 0:
                    state_dict = {}
                    with safe_open(
                        filename=os.path.join(
                            model_save_file,
                            f"n_layer_{n_layer}_lr_{lr}_d_model_{d_model}_best.safetensors",
                        ),
                        framework="pt",
                        device=device,
                    ) as f:
                        model_dict = {}
                        optimizer_dict = {}
                        for key in model.state_dict().keys():
                            model_dict[key] = f.get_tensor(key)
                        model.load_state_dict(model_dict)
                    optimizer_state = torch.load(
                        os.path.join(
                            model_save_file,
                            f"n_layer_{n_layer}_lr_{lr}_d_model_{d_model}_best_optimizer.pth",
                        )
                    )
                    optimizer.load_state_dict(optimizer_dict["optimizer_state"])
                    epoch = optimizer_state["epoch"]
                    model.load_state_dict(state_dict["model"])
                    if train_args.resume:
                        optimizer = get_optimizer(model=model, train_args=train_args)
                    # optimizer.load_state_dict(state_dict['optimizer_config'])
                    run_logger.info(
                        f"use model_optimizer_state_lr_{lr}_epoch_{start_epoch}.pth to train"
                    )
                # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=epochs,eta_min=0)
                train_args.max_steps = epochs
                lr_scheduler = get_scheduler(optimizer=optimizer, train_args=train_args)
                best_model = None
                best_loss = math.inf
                best_epoch = None
                train_loss_history = []
                model.train()
                run_logger.info(
                    f"the length of train_set_loader:{len(train_set_loader)}"
                )

                for epoch in tqdm(
                    range(start_epoch, epochs),
                    initial=start_epoch,
                    total=epochs - start_epoch,
                    desc=f"training_model_{n_layers} layers_lr_{lr}_d_model_{d_model}",
                ):
                    total_iter = 0
                    k = 0
                    end_cost = 0.0

                    for x, y, depth in tqdm(
                        train_set_loader,
                        total=len(train_set_loader),
                        desc=f"{n_layers}_layers_lr_{lr}_d_model_{d_model}_train_epoch_{epoch}",
                    ):
                        run_logger.debug(f"the shape of x:{x.shape}")
                        run_logger.debug(f"the shape of y:{y.shape}")
                        x = x.unsqueeze(dim=-1).to(device)
                        y = y.to(device)
                        pred = model(x)
                        run_logger.debug(
                            f"the shape of pred.squeeze:{pred.squeeze().shape}"
                        )
                        L = loss(y, pred.squeeze()) * 512

                        optimizer.zero_grad()

                        L.backward()
                        optimizer.step()
                        total_iter += 1
                        end_cost = L.item()
                    train_loss_history.append(end_cost)
                    # if total_iter%iter_num==0 and len(train_set_loader)>5:
                    #     k+=1
                    #     writer.add_scalar(tag=f'train_split_num_{split_num}_n_layers_{n_layers}_lr_{lr}_batch_size_{batch_size}_start_epoch_{start_epoch}_train_epoch_{epoch}_iter',scalar_value=L.item(),global_step=k)
                    writer.add_scalar(
                        tag=f"n_layer_{n_layer}_lr_{lr}_d_model_{d_model}_loss",
                        scalar_value=end_cost,
                        global_step=epoch,
                    )

                    lr_scheduler.step(step=epoch)

                    # save_dict={'model':model.state_dict(),
                    #            'optimizer_config':optimizer.state_dict(),
                    #            'epoch':epoch
                    #            }
                    # run_logger.info(f'saving the model_optimizer_epoch config...')
                    # torch.save(save_dict,os.path.join(model_save_file,f'with_embedding_{with_embedding}_model_optimizer_state_out_channels_{out_channels}_split_num_{split_num}_n_layers_{n_layers}_lr_{lr}_batch_size{batch_size}_epoch_{epoch}.pth'))

                    model.eval()
                    l = 0.0
                    for x, y, depth in tqdm(
                        test_set_loader, total=len(test_set_loader), desc=f"test_model"
                    ):
                        run_logger.debug(f"the shape of x:{x.shape}")
                        with torch.no_grad():
                            x = x.unsqueeze(dim=-1).to(device)
                            y = y.to(device)
                            pred = model(x)
                            run_logger.debug(f"pred shape:{pred.shape}")
                            L = loss(y, pred.squeeze()) * 512
                            l = l + L.item() * x.shape[0]

                    if best_loss > l:
                        best_model = model
                        best_epoch = epoch
                        # if (epoch+1)%train_args.save_epoch_freq==0:
                        # save_dict = {'model': model.state_dict(),
                        #              'optimizer_config': optimizer.state_dict(),
                        #              'epoch': epoch
                        #              }
                        # best_epoch=epoch
                        # best_loss=l
                        # run_logger.info(f'saving the best model,model_optimizer_epoch config...')
                        # save_file(tensors=save_dict, filename=os.path.join(model_save_file,
                        #                                    f'n_layer_{n_layer}_lr_{lr}_d_model_{d_model}_epoch{epoch}_best.safetensors'))
                        state_dict = model.state_dict()
                        run_logger.debug(
                            f"the type of mode.state_dict():{type(state_dict)}"
                        )
                        # Flatten the state_dict to get tensors only
                        run_logger.debug(
                            f"the type of optimizer.state_dict():{type(optimizer.state_dict())}"
                        )
                        # run_logger.debug(f"the type of optimizer.state_dict():{type()}")
                        opimizer_state = {
                            "opoch": epoch,
                            "optimizer_state": optimizer.state_dict(),
                        }
                        torch.save(
                            obj=optimizer,
                            f=os.path.join(
                                model_save_file,
                                f"n_layer_{n_layer}_lr_{lr}_d_model_{d_model}_best_optimizer.pth",
                            ),
                        )
                        save_file(
                            tensors=state_dict,
                            filename=os.path.join(
                                model_save_file,
                                f"n_layer_{n_layer}_lr_{lr}_d_model_{d_model}_best.safetensors",
                            ),
                        )

                train_loss_history_array = np.array(train_loss_history)
                train_loss_history_file = os.path.join(
                    train_loss_history_path,
                    f"n_layer_{n_layer}_lr_{lr}_d_model_{d_model}_best.npy",
                )
                np.save(train_loss_history_file, train_loss_history_array)

                # for test
                fig_save_path = os.path.join(
                    data_args.data_save_path, "eRank_test_figs"
                )
                if not os.path.exists(fig_save_path):
                    os.mkdir(fig_save_path)
                fig_n_layer_save_path = os.path.join(
                    fig_save_path, f"n_layer_{n_layer}"
                )
                if not os.path.exists(fig_n_layer_save_path):
                    os.mkdir(fig_n_layer_save_path)
                fig_n_layer_channel_save_path = os.path.join(
                    fig_n_layer_save_path, f"d_model_{model_args.d_model}"
                )
                if not os.path.exists(fig_n_layer_channel_save_path):
                    os.mkdir(fig_n_layer_channel_save_path)
                fig_n_layer_channel_lr_save_path = os.path.join(
                    fig_n_layer_channel_save_path, f"learning_rate_{train_args.lr}"
                )
                if not os.path.exists(fig_n_layer_channel_lr_save_path):
                    os.mkdir(fig_n_layer_channel_lr_save_path)

                l = 0.0
                num_iter = 0
                test_index = dataset.index_mapping["index"]["test_set_index"]
                index2str = dataset.index_mapping["ind2str"]
                run_logger.debug(index2str)
                run_logger.debug(f"the len of val_set_loader:{len(val_set_loader)}")
                model = best_model
                model.eval()
                for x, y, depth in tqdm(
                    test_set_loader_for_save_fig,
                    total=len(test_set_loader),
                    desc=f"test_model",
                ):
                    run_logger.debug(f"the shape of x:{x.shape}")
                    x = x.unsqueeze(dim=-1).to(device)
                    y = y.to(device)
                    pred = model(x).squeeze()
                    run_logger.debug(f"pred shape:{pred.shape}")
                    str_name = index2str[f"{test_index[num_iter]}"]
                    num_iter += 1

                    fig, axs = plt.subplots()
                    # run_logger.debug(y.data.numpy())
                    axs.plot(y.cpu().data.numpy().flatten(), label="True")
                    axs.plot(pred.cpu().data.numpy().flatten(), label="fake")
                    run_logger.debug(pred.cpu().data.numpy())
                    axs.set_title(f"{str_name}")
                    axs.legend()
                    if (num_iter + 1) % train_args.saving_fig_log_freq == 0:
                        run_logger.info(f"saving the test_fig...")
                    plt.savefig(
                        os.path.join(
                            fig_n_layer_channel_lr_save_path,
                            f"start_epoch_{start_epoch}_{str_name}_best_epoch_{best_epoch}.jpg",
                        )
                    )
                    plt.close()

    writer.close()

    return (
        DataLoader(
            MyDataset(dataset.datasets["test_set"], dataset.datasets["test_depth_set"])
        ),
        model,
    )


if __name__ == "__main__":
    run_logger.setLevel(logging.INFO)
    args_parser = HfArgumentParser([TrainArgs, DataArgs, ModelArgs])
    train_args, data_args, model_args = args_parser.parse_args_into_dataclasses()

    test_data_loader, model = run(
        train_args=train_args, data_args=data_args, model_args=model_args
    )
