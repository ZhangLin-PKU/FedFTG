import os
import torch
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.distributions import Dirichlet, Categorical, LogNormal
import random

from typing import List
plt.style.use('seaborn')

__all__ = ['dataPrep']

def load_from_list(file_path):
    img_list = []
    lbl_list = []
    for line in open(file_path).readlines():
        img_name, img_lbl = line.strip().split()
        img_list.append(img_name)
        lbl_list.append(int(img_lbl))
    return img_list, lbl_list

class dataPrep:
    def __init__(self,
                 dataset_name: str,
                 root_dir: Path) -> None:

        self.root_dir = root_dir
        self.dataset_name = dataset_name
        if self.dataset_name == "CIFAR10":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                                                 std=[0.247, 0.243, 0.262])])

            self.train_data = datasets.CIFAR10(root_dir / "Raw/",
                                      train=True, download=True, transform=transform)
            self.test_data = datasets.CIFAR10(root_dir / "Raw/",
                                      train=False, download=True, transform=transform)
            self.num_train_data = len(self.train_data)
            self.num_classes = 10
        elif self.dataset_name == "CIFAR100":
            
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                 std=[0.2675, 0.2565, 0.2761])])
            self.train_data = datasets.CIFAR100(root_dir / "Raw/",
                                      train=True, download=True, transform=transform)
            self.test_data = datasets.CIFAR100(root_dir / "Raw/",
                                      train=False, download=True, transform=transform)

            self.num_train_data = len(self.train_data)
            self.num_classes = 100
        else:
            raise ValueError("Unknown dataset_name")

    def make(self,
             mode: int,
             num_clients: int,
             show_plots: bool = False,
             **kwargs) -> None:

        # save_name = '{}_{}_{}_{}_{}'.format(self.dataset_name, num_clients, kwargs.get('seed'), 'Dirichlet', str(kwargs.get('dir_alpha')).ljust(5))

        save_name = "%s_%d_%d_%s_%s" %  (self.dataset_name, num_clients, kwargs.get('seed'), 'Dirichlet', '%.3f' % kwargs.get('dir_alpha'))
        np.random.seed(kwargs.get('seed'))
        torch.manual_seed(kwargs.get('seed'))
        random.seed(kwargs.get('seed'))

        if os.path.exists(self.root_dir / save_name):
            shutil.rmtree(self.root_dir / save_name)
        client_data_path = Path(self.root_dir / save_name)
        client_data_path.mkdir()

        # if not isinstance(self.test_data.targets, torch.Tensor):
        #     self.test_data.targets = torch.tensor(self.test_data.targets)
        # test_data = [self.test_data[j] for j in range(len(self.test_data))]
        # torch.save(test_data,
        #            client_data_path / "test_data.pth")

        if mode == 0:       # IID
            # Shuffle data
            data_ids = torch.randperm(self.num_train_data, dtype=torch.int32)
            num_data_per_client = self.num_train_data// num_clients

            if not isinstance(self.train_data.targets, torch.Tensor):
                self.train_data.targets = torch.tensor(self.train_data.targets)

            pbar = tqdm(range(num_clients), desc = f"{self.dataset_name} IID: ")
            for i in pbar:
                client_path = Path(client_data_path / str(i))
                client_path.mkdir()

                # TODO: Make this parallel for large number of clients & large datasets (Maybe not required)
                train_data = [self.train_data[j]
                              for j in data_ids[i*num_data_per_client: (i+1)*num_data_per_client]]

                pbar.set_postfix({'# data / Client': num_data_per_client})

                if show_plots:
                    self._plot(train_data, title=f"Client {i+1} Data Distribution")

                # Split data equally and send to the client
                torch.save(train_data,
                           client_data_path / str(i) / "data.pth")
        elif mode == 1:     # Non IID Balanced
            num_data_per_client = self.num_train_data//num_clients
            classs_sampler = Dirichlet(torch.empty(self.num_classes).fill_(kwargs.get('dir_alpha')))
            # print(torch.empty(self.num_classes).fill_(2.0))
            if not isinstance(self.train_data.targets, torch.Tensor):
                self.train_data.targets = torch.tensor(self.train_data.targets)

            assigned_ids = []
            pbar = tqdm(range(num_clients), desc = f"{self.dataset_name} Non-IID Balanced: ")
            clnt_x = []
            clnt_y = []

            for i in pbar:

                # Compute class prior probabilities for each client
                p_ij = classs_sampler.sample()  # Share of jth class for ith client (always sums to 1)
                # print(p_ij)
                weights = torch.zeros(self.num_train_data)
                # print(torch.nonzero(self.train_data.targets == 9))
                for c_id in range(self.num_classes):
                    weights[self.train_data.targets == c_id] = p_ij[c_id]
                weights[assigned_ids] = 0.0 # So that previously assigned data are not sampled again

                # Sample each data point uniformly without replacement based on
                # the sampling probability assigned based on its class
                data_ids = torch.multinomial(weights, num_data_per_client, replacement=False)

                train_data = [self.train_data[j] for j in data_ids]
                train_data_x = [data_[0].numpy() for data_ in train_data]
                train_data_y = [data_[1] for data_ in train_data]
                # print(f"Client {i} has {len(train_data)} data points.")
                pbar.set_postfix({'# data / Client': len(train_data)})

                assigned_ids += data_ids.tolist()

                clnt_x.append(np.array(train_data_x).astype(np.float32))
                # print(clnt_x[-1].shape)
                clnt_y.append(np.array(train_data_y).astype(np.int64).reshape(-1, 1))

                if show_plots:
                    self._plot(train_data, title=f"Client {i+1} Data Distribution")

            tst_load = torch.utils.data.DataLoader(self.test_data, batch_size=10000, shuffle=False, num_workers=1)
            tst_x, tst_y = next(iter(tst_load))

            tst_x = tst_x.numpy()
            tst_y = tst_y.numpy().reshape(-1, 1)
            clnt_x = np.asarray(clnt_x)
            clnt_y = np.asarray(clnt_y)

            np.save(client_data_path / 'clnt_x.npy', clnt_x)
            np.save(client_data_path / 'clnt_y.npy', clnt_y)
            np.save(client_data_path / 'tst_x.npy', tst_x)
            np.save(client_data_path / 'tst_y.npy', tst_y)

        elif mode == 2:     # Non IID Unbalanced
            num_data_per_client = self.num_train_data // num_clients
            num_data_per_class = self.num_train_data / (self.num_classes * num_clients)
            classs_sampler = Dirichlet(torch.empty(self.num_classes).fill_(kwargs.get('dir_alpha')))

            assigned_ids = []
            pbar = tqdm(range(num_clients), desc = f"{self.dataset_name} Non-IID Unbalanced: ")

            if not isinstance(self.train_data.targets, torch.Tensor):
                self.train_data.targets = torch.tensor(self.train_data.targets)

            for i in pbar:
                train_data = []
                client_path = Path(client_data_path / str(i))
                client_path.mkdir()
                # Compute class prior probabilities for each client
                p_ij = classs_sampler.sample()  # Share of jth class for ith client (always sums to 1)
                c_sampler = Categorical(p_ij)
                data_sampler = LogNormal(torch.tensor(num_data_per_class).log(),
                                         kwargs.get('lognorm_std'))

                while(True):
                    num_data_left = num_data_per_client - len(train_data)
                    c = c_sampler.sample()
                    num_data_c = int(data_sampler.sample())
                    # print(c, num_data_c, len(train_data))
                    data_ids = torch.nonzero(self.train_data.targets == c.item()).flatten()
                    # data_ids = [x for x in data_ids if x not in assigned_ids] # Remove duplicated ids
                    # print(data_ids.shape)
                    num_data_c = min(num_data_c, data_ids.shape[0])
                    if num_data_c >= num_data_left :
                        train_data += [self.train_data[j] for j in data_ids[:num_data_left]]
                        break
                    else:
                        train_data += [self.train_data[j] for j in data_ids[:num_data_c]]
                        assigned_ids += data_ids[:num_data_c].tolist()

                pbar.set_postfix({'# data / Client': len(train_data)})
                torch.save(train_data,
                           client_data_path / str(i) / "data.pth")
                if show_plots:
                    self._plot(train_data, title=f"Client {i+1} Data Distribution")

        else:
            raise ValueError("Unknown mode. Mode must be {0,1}")

    def _plot(self, data: List, title: str = None) -> None:
        labels = [int(d[1]) for d in data]
        # print(labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(labels, bins=np.arange(self.num_classes + 1) - 0.5)
        ax.set_xticks(range(self.num_classes))
        ax.set_xlim([-1, self.num_classes])
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('Class ID', fontsize=13)
        ax.set_ylabel('# samples', fontsize=13)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    d = dataPrep("CIFAR10", root_dir =Path("results/Data/"))
    d.make(1, 100, dir_alpha=0.05, lognorm_std=0.0, show_plots=False, seed=1024)

    # from torch.utils.data import DataLoader
    # dir_ = Path("Data/client_data")
    # data =torch.load(dir_/ "0/data.pth")
    # dataloader = DataLoader(data, shuffle=True, num_workers=4, batch_size=2)
    # print(next(iter(dataloader)))

