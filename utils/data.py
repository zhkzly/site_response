# user zhengkelong
import matplotlib.pyplot as plt
import numpy
import numpy as np
import os
import pandas as pd
import math
import json
from tqdm import tqdm

import logging

data_logger = logging.getLogger("data_logger")
data_hander = logging.StreamHandler()
fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s-%(lineno)s"
data_logger_formater = logging.Formatter(fmt=fmt)
data_hander.setFormatter(data_logger_formater)
data_logger.addHandler(data_hander)
data_logger.setLevel(logging.INFO)

file_path = "/media/zkl/zkl_T7/observe_prediction"

import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, dataset, depth_set, dtype=torch.float32):
        # train_set
        if isinstance(dataset, np.ndarray):
            self.dataset = torch.from_numpy(np.array(dataset)).type(dtype)
            self.depth_set = torch.from_numpy(np.array(depth_set)).long()
        else:
            self.depth_set = depth_set.long()
            self.dataset = dataset.type(dtype)

    def __len__(self):
        return self.dataset.shape[1]

    def __getitem__(self, item):
        return self.dataset[0][item], self.dataset[1][item], self.depth_set[item]


class Data:
    # 划分数据集，可以进行一定的预处理，数据集可以采用预先加载的形式，并且也要保留对应的数据集的标签索引，比如第一行代表哪一个台站
    #
    def __init__(
        self,
        file_path=file_path,
        data_save_path="./modified_model/datas",
        loading_from_file=None,
        index_mapping_file=None,
        ratio=(7, 0, 3),
        seed=123,
    ):
        self.file_path = file_path
        self.ratio = ratio
        self.data_save_path = data_save_path
        np.random.seed(seed)
        # data_set = {'train_set': np.stack([inputs[train_set_index], labels[train_set_index]], axis=0),
        # 'val_set': np.stack([inputs[train_set_index], labels[train_set_index]], axis=0),
        # 'test_set': np.stack([inputs[train_set_index], labels[train_set_index]], axis=0),
        # 'train_depth_set': depth_label[train_set_index],
        # 'val_depth_set': depth_label[val_set_index],
        # 'test_depth_set': depth_label[test_set_index]}

        # index_ = {'train_set_index': train_set_index.tolist(), 'val_set_index': val_set_index.tolist(),
        #           'test_set_index': test_set_index.tolist()}

        self.datasets, self.index_mapping = self._loading_data(
            file_path, loading_from_file, index_mapping_file, ratio
        )
        self.train_len = self.datasets["train_set"].shape[1]

    def _loading_data(
        self,
        file_path,
        loading_from_file=None,
        index_mapping_file=None,
        ratio=(7, 0, 3),
        save=True,
    ):
        # 采用npz的形式保存数据，采用json的形式保存映射关系
        if loading_from_file is not None:
            index_mapping = None
            with open(index_mapping_file, "r") as fp:
                index_mapping = json.load(fp)
            dict_like = np.load(loading_from_file, allow_pickle=True)
            dataset = {key: value for key, value in dict_like.items()}
            data_logger.debug(dataset.keys())
            return dataset, index_mapping
        else:
            path = os.path.abspath(file_path)
            data_logger.debug(f"path:{path}")
            paths = sorted(os.listdir(path))
            total_len = len(paths)
            index_ = np.arange(0, total_len)
            np.random.shuffle(index_)
            train_set_index = index_[: math.ceil(ratio[0] * total_len / 10)]
            data_logger.info(f"the shape of train_set_index:{train_set_index.shape}")
            val_set_index = index_[
                math.ceil(ratio[0] * total_len / 10) : math.ceil(
                    total_len * (ratio[0] + ratio[1]) / 10
                )
            ]
            data_logger.info(f"the shape of val_set_index:{val_set_index.shape}")
            test_set_index = index_[math.ceil((ratio[0] + ratio[1]) / 10 * total_len) :]
            data_logger.info(f"the shape of test_set_index:{test_set_index.shape}")

            def _get_depth_label(datas):
                N = datas.shape[0]
                for i in range(N):
                    if 0 <= datas[i] < 200:
                        datas[i] = 0
                    elif 200 <= datas[i] < 300:
                        datas[i] = 1
                    else:
                        datas[i] = 2

                return datas

            str2ind = {path.split(".")[0]: i for i, path in enumerate(paths)}
            ind2str = {i: path.split(".")[0] for i, path in enumerate(paths)}
            data_logger.debug(ind2str)
            datalist = []
            str2depth = {}
            depthlist = []
            for _path in tqdm(paths, total=total_len, desc="for loading excel"):
                # data_logger.debug(f'{os.path.join(file_path, _path)}')
                data_logger.debug(paths[0])
                data_logger.debug(
                    "++++++++++++++++++++++++++++++++++++++++++\n+++++++++++++++++++++++++++++++++++++++++"
                )
                data_frame_dict = pd.read_excel(
                    os.path.join(file_path, _path),
                    header=None,
                    sheet_name=[0, 1],
                    engine="openpyxl",
                )
                datalist.append(data_frame_dict[0].iloc[:, 1:3].to_numpy().T)
                depth = math.ceil(data_frame_dict[1].iloc[0, 0])
                # data_logger.debug(f'depth:{depth}')
                depthlist.append(depth)
                str2depth[_path.split(".")[0]] = depth

            depth_label = _get_depth_label(np.array(depthlist, dtype=np.int32))
            data_logger.debug(f"{depth_label}")
            depth_array = np.array(depthlist, dtype=np.int32)
            data_logger.debug(f"depth label:{depth_label[0]}")

            data_logger.debug(f"datalist:{datalist[0]}")
            labels, inputs = [np.array(data) for data in zip(*datalist)]
            data_logger.debug(f"shape of labels:{labels.shape}")
            data_logger.debug(f"labels:{labels[0]}")
            data_logger.debug(f"inputs:{inputs[0]}")
            data_set = {
                "train_set": np.stack(
                    [inputs[train_set_index], labels[train_set_index]], axis=0
                ),
                "val_set": np.stack(
                    [inputs[val_set_index], labels[val_set_index]], axis=0
                ),
                "test_set": np.stack(
                    [inputs[test_set_index], labels[test_set_index]], axis=0
                ),
                "train_depth_set": depth_label[train_set_index],
                "val_depth_set": depth_label[val_set_index],
                "test_depth_set": depth_label[test_set_index],
            }
            index_mapping = None
            if save:
                if not os.path.exists(self.data_save_path):
                    os.mkdir(self.data_save_path)
                # if not os.path.exists(save_excel_path):
                #     os.mkdir(save_path)
                #
                # save_excel_train=os.path.join(os.path.dirname(file_path),'data','origin_data','train')
                # if not os.path.exists(save_excel_train):
                #     os.mkdir(save_excel_train)
                # save_excel_val=os.path.join(os.path.dirname(file_path),'data','origin_data','val')
                # if not os.path.exists(save_excel_val):
                #     os.mkdir(save_excel_val)
                # save_excel_test=os.path.join(os.path.dirname(file_path),'data','origin_data','test')
                # if not os.path.exists(save_excel_test):
                #     os.mkdir(save_excel_test)
                #
                data_logger.debug(paths[0])
                data_logger.debug(
                    "++++++++++++++++++++++++++++++++++++++++++\n+++++++++++++++++++++++++++++++++++++++++"
                )
                for i, _path in enumerate(
                    tqdm(paths, total=total_len, desc="for loading excel")
                ):
                    # data_logger.debug(f'{os.path.join(file_path, _path)}')
                    data_frame_dict = pd.read_excel(
                        os.path.join(file_path, _path), header=None, sheet_name=[0, 1]
                    )

                    datalist.append(data_frame_dict[0].iloc[:, 1:3].to_numpy().T)
                    depth = math.ceil(data_frame_dict[1].iloc[0, 0])
                    # data_logger.debug(f'depth:{depth}')
                    depthlist.append(depth)
                    str2depth[_path.split(".")[0]] = depth

                data_save_path = os.path.join(self.data_save_path, "train_val_test")
                data_logger.info(f"save numpy data to {data_save_path}")
                np.savez(data_save_path, **data_set)
                index_path = os.path.join(self.data_save_path, "mapping.json")
                data_logger.info(f"save mapping index to {index_path}")
                index_ = {
                    "train_set_index": train_set_index.tolist(),
                    "val_set_index": val_set_index.tolist(),
                    "test_set_index": test_set_index.tolist(),
                }
                index_mapping = {
                    "str2ind": str2ind,
                    "ind2str": ind2str,
                    "str2depth": str2depth,
                    "index": index_,
                }
                with open(index_path, mode="w") as fp:
                    json.dump(index_mapping, fp)

        return data_set, index_mapping

    def get_dataloaders(self, batch_size=128, dtype=torch.float32, shuffle=True):
        return (
            DataLoader(
                MyDataset(self.datasets["train_set"], self.datasets["train_depth_set"]),
                batch_size=batch_size,
                shuffle=shuffle,
            ),
            DataLoader(
                MyDataset(self.datasets["val_set"], self.datasets["val_depth_set"]),
                batch_size=32,
            ),
            DataLoader(
                MyDataset(self.datasets["test_set"], self.datasets["test_depth_set"]),
                batch_size=batch_size,
            ),
        )


def check_depth(file_path):
    path = os.path.abspath(file_path)
    paths = list(sorted(os.listdir(path)))
    total_len = len(paths)
    # sheetname = None # 读取全部表，得到 OrderDict：key为表名，value为 DataFrame
    datas = {
        path: math.ceil(
            pd.read_excel(
                os.path.join(file_path, path), header=None, sheet_name=1
            ).iloc[0, 0]
        )
        for path in paths
    }
    data_logger.setLevel(logging.INFO)
    data_logger.debug(datas.values())
    bt0_100 = []
    bt100_200 = []
    bt200_300 = []
    bt300_400 = []
    bt400_500 = []
    bt500_600 = []
    bt600_700 = []
    others = []
    for value in datas.values():
        if 0 <= value < 100:
            bt0_100.append(value)
        elif 100 <= value < 200:
            bt100_200.append(value)
        elif 200 <= value < 300:
            bt200_300.append(value)
        elif 300 <= value < 400:
            bt300_400.append(value)
        elif 400 <= value < 500:
            bt400_500.append(value)
        elif 500 <= value < 600:
            bt500_600.append(value)
        elif 600 <= value < 700:
            bt600_700.append(value)
        else:
            others.append(value)
    # 0-100:0,[]
    print(f"len 0-100:{len(bt0_100)},{sorted(bt0_100)}")
    # 100-200:447
    # [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    # 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    # 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    # 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    # 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    # 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    # 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    # 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    # 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
    # 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
    # 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 102, 102, 102, 102, 102, 102, 102, 102,
    # 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 102, 103, 103, 103, 103, 103, 103,
    # 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 104, 104, 104, 104, 104, 104, 104, 105, 105, 105, 105,
    # 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106,
    # 106, 106, 107, 107, 107, 107, 107, 107, 107, 107, 108, 108, 109, 109, 109, 109, 110, 110, 110, 110, 110, 110, 110,
    # 110, 110, 110, 110, 111, 111, 111, 112, 112, 112, 112, 112, 113, 113, 114, 114, 115, 115, 115, 115, 116, 116, 117,
    # 117, 117, 117, 118, 120, 120, 120, 120, 120, 122, 122, 122, 122, 122, 122, 122, 123, 124, 124, 127, 127, 127, 127,
    # 128, 132, 135, 135, 136, 137, 140, 140, 142, 143, 144, 145, 147, 147, 147, 147, 148, 148, 150, 150, 150, 150, 150,
    # 150, 150, 150, 150, 150, 150, 150, 150, 151, 152, 152, 153, 153, 154, 155, 156, 157, 157, 158, 160, 160, 162, 165,
    # 168, 170, 172, 177, 177, 177, 180, 180, 180, 197]
    print(f"len 100-200:{len(bt100_200)},{sorted(bt100_200)}")
    # 200 - 300: 143
    # [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    # 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    # 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    # 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    # 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
    # 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201, 201,
    # 201, 201, 201, 201, 201, 202, 202, 202, 202, 202, 203, 203, 203, 203,
    # 203, 203, 203, 204, 204, 204, 205, 205, 205, 205, 206, 207, 207, 207,
    # 207, 207, 207, 207, 208, 208, 209, 209, 210, 210, 210, 212, 212, 214, 216,
    # 218, 220, 220, 221, 222, 222, 227, 228, 231, 237, 239, 241, 247, 250, 251, 252,
    # 252, 255, 260, 263, 268]
    print(f"len 200-300:{len(bt200_300)},{sorted(bt200_300)}")
    # 300-400:26
    # [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300,
    # 300, 300, 302, 303, 304, 306, 312, 330, 337, 350, 360, 360, 387]
    print(f"len 300-400:{len(bt300_400)},{sorted(bt300_400)}")
    # 400-500:5
    # [402, 437, 400, 400, 450]
    print(f"len 400-500:{len(bt400_500)},{sorted(bt400_500)}")
    # 500-600:4
    # [504, 510, 526, 562]
    print(f"len 500-600:{len(bt500_600)},{sorted(bt500_600)}")
    # 600-700:1
    # [629]
    print(f"len 600-700:{len(bt600_700)},{sorted(bt600_700)}")
    # >=700:12
    # [705, 709, 802, 822, 923, 983, 996, 1055, 1206, 1509, 2000, 2003]
    print(f"len other:{len(others)},{sorted(others)}")


if __name__ == "__main__":
    # check_depth(r'E:\观测与预测_512')
    data_logger.setLevel(logging.INFO)
    file_path = "/media/zkl/zkl_T7/preprocession/observe_prediction_512"
    data = Data(file_path=file_path)
