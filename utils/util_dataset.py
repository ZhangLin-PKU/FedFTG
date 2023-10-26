import os
import numpy as np
import torchvision
import torch
from torch.utils import data
import torchvision.transforms as transforms
from scipy import io
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

COLOR_MAP = ['red', 'green', 'blue', 'black', 'brown', 'purple', 'yellow', 'pink', 'cyan', 'gray']


class DatasetObject:
    def __init__(self, dataset, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path=''):
        self.dataset = dataset
        self.n_client = n_client
        # self.rule = rule
        # self.rule_arg = rule_arg
        # self.seed = seed
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        self.name = "%s_%d_%d_%s_%s" % (self.dataset, self.n_client, seed, rule, rule_arg_str)
        self.name += '_%f' % unbalanced_sgm if unbalanced_sgm != 0 else ''
        # self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.channels, self.width, self.height, self.n_cls = self._get_data_info()

        self.clnt_x, self.clnt_y, self.tst_x, self.tst_y = self._load_split_data(seed, rule, rule_arg, unbalanced_sgm)

        # self.set_data()

    def _get_data_info(self):
        if self.dataset == 'CIFAR10':
            return [3, 32, 32, 10]
        elif self.dataset == 'CIFAR100':
            return [3, 32, 32, 100]
        else:
            raise ValueError('Wrong dataset.')

    def _load_data(self):
        if self.dataset == 'CIFAR10':
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                                                 std=[0.247, 0.243, 0.262])])

            trnset = torchvision.datasets.CIFAR10(root='%sData/Raw' % self.data_path,
                                                  train=True, download=True, transform=transform)
            tstset = torchvision.datasets.CIFAR10(root='%sData/Raw' % self.data_path,
                                                  train=False, download=True, transform=transform)

            trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
            tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
            trn_x, trn_y = next(iter(trn_load))
            tst_x, tst_y = next(iter(tst_load))

            trn_x = trn_x.numpy()
            trn_y = trn_y.numpy().reshape(-1, 1)
            tst_x = tst_x.numpy()
            tst_y = tst_y.numpy().reshape(-1, 1)
            return trn_x, trn_y, tst_x, tst_y

        elif self.dataset == 'CIFAR100':
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                 std=[0.2675, 0.2565, 0.2761])])
            trnset = torchvision.datasets.CIFAR100(root='%sData/Raw' % self.data_path,
                                                   train=True, download=True, transform=transform)
            tstset = torchvision.datasets.CIFAR100(root='%sData/Raw' % self.data_path,
                                                   train=False, download=True, transform=transform)

            trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=0)
            tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=0)
            trn_x, trn_y = next(iter(trn_load))
            tst_x, tst_y = next(iter(tst_load))

            trn_x = trn_x.numpy()
            trn_y = trn_y.numpy().reshape(-1, 1)
            tst_x = tst_x.numpy()
            tst_y = tst_y.numpy().reshape(-1, 1)
            return trn_x, trn_y, tst_x, tst_y
        else:
            raise ValueError('Wrong dataset')

    def _split_data(self, clnt_data_list, trn_x, trn_y, rule, rule_arg, sgm):
        if rule == 'Dirichlet':
            cls_priors = np.random.dirichlet(alpha=[rule_arg] * self.n_cls, size=self.n_client)
            prior_cumsum = np.cumsum(cls_priors, axis=1)

            idx_list = [np.where(trn_y == i)[0] for i in range(self.n_cls)]
            cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]

            clnt_x = [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                      for clnt__ in range(self.n_client)]
            clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)]

            while np.sum(clnt_data_list) != 0:
                curr_clnt = np.random.randint(self.n_client)
                # If current node is full resample a client
                # print('Remaining Data: %d' %np.sum(clnt_data_list))
                if clnt_data_list[curr_clnt] <= 0:
                    continue
                clnt_data_list[curr_clnt] -= 1
                curr_prior = prior_cumsum[curr_clnt]
                while True:
                    cls_label = np.argmax(np.random.uniform() <= curr_prior)
                    # Redraw class label if trn_y is out of that class
                    if cls_amount[cls_label] <= 0:
                        continue
                    cls_amount[cls_label] -= 1

                    clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                    clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]

                    break

            clnt_x = np.asarray(clnt_x)
            clnt_y = np.asarray(clnt_y)

            cls_means = np.zeros((self.n_client, self.n_cls))
            for clnt in range(self.n_client):
                for cls in range(self.n_cls):
                    cls_means[clnt, cls] = np.mean(clnt_y[clnt] == cls)
            prior_real_diff = np.abs(cls_means - cls_priors)
            print('--- Max deviation from prior: %.4f' % np.max(prior_real_diff))
            print('--- Min deviation from prior: %.4f' % np.min(prior_real_diff))
            return clnt_x, clnt_y

        elif rule == 'iid':
            if self.dataset == 'CIFAR100_' and sgm == 0:
                assert len(trn_y) // 100 % self.n_client == 0
                # create perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(trn_y[:, 0])
                n_data_per_clnt = len(trn_y) // self.n_client
                # clnt_x dtype needs to be float32, the same as weights
                clnt_x = np.zeros((self.n_client, n_data_per_clnt, 3, 32, 32), dtype=np.float32)
                clnt_y = np.zeros((self.n_client, n_data_per_clnt, 1), dtype=np.float32)
                trn_x = trn_x[idx]  # 50000*3*32*32
                trn_y = trn_y[idx]
                n_cls_sample_per_device = n_data_per_clnt // 100
                for i in range(self.n_client):  # devices
                    for j in range(100):  # class
                        clnt_x[i, n_cls_sample_per_device * j:n_cls_sample_per_device * (j + 1), :, :, :] = \
                            trn_x[500 * j + n_cls_sample_per_device * i:500 * j + n_cls_sample_per_device * (i + 1), :, :, :]
                        clnt_y[i, n_cls_sample_per_device * j:n_cls_sample_per_device * (j + 1), :] = \
                            trn_y[500 * j + n_cls_sample_per_device * i:500 * j + n_cls_sample_per_device * (i + 1), :]
            else:

                clnt_x = [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                          for clnt__ in range(self.n_client)]
                clnt_y = [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)]

                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_client):
                    clnt_x[clnt_idx_] = trn_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_ + 1]]
                    clnt_y[clnt_idx_] = trn_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_ + 1]]

                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)
            return clnt_x, clnt_y

        else:
            raise ValueError('Wrong data segmentation rule')

    def _load_split_data(self, seed, rule, rule_arg, sgm):
        # Prepare data if not ready
        clnt_x = []
        clnt_y = []
        tst_x = []
        tst_y = []
        if not os.path.exists('%sData/%s' % (self.data_path, self.name)):
            print()
            trn_x, trn_y, tst_x, tst_y = self._load_data()
            # Shuffle Data
            np.random.seed(seed)
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]

            # Utilize lognorm to obtain balanced/unbalanced data distribution
            n_data_per_clnt = int((len(trn_y)) / self.n_client)
            # Draw from lognormal distribution
            clnt_data_list = (
                np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=sgm, size=self.n_client))
            clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(trn_y)).astype(int)
            diff = np.sum(clnt_data_list) - len(trn_y)

            # Add/Subtract the excess number starting from first client
            if diff != 0:
                for clnt_i in range(self.n_client):
                    if clnt_data_list[clnt_i] > diff:
                        clnt_data_list[clnt_i] -= diff
                        break
            ################

            clnt_x, clnt_y = self._split_data(clnt_data_list, trn_x, trn_y, rule, rule_arg, sgm)

            # Save data
            os.mkdir('%sData/%s' % (self.data_path, self.name))

            np.save('%sData/%s/clnt_x.npy' % (self.data_path, self.name), clnt_x)
            np.save('%sData/%s/clnt_y.npy' % (self.data_path, self.name), clnt_y)

            np.save('%sData/%s/tst_x.npy' % (self.data_path, self.name), tst_x)
            np.save('%sData/%s/tst_y.npy' % (self.data_path, self.name), tst_y)

        else:
            print("Data is already downloaded")
            clnt_x = np.load('%sData/%s/clnt_x.npy' % (self.data_path, self.name),  allow_pickle=True)
            clnt_y = np.load('%sData/%s/clnt_y.npy' % (self.data_path, self.name),  allow_pickle=True)

            tst_x = np.load('%sData/%s/tst_x.npy' % (self.data_path, self.name),  allow_pickle=True)
            tst_y = np.load('%sData/%s/tst_y.npy' % (self.data_path, self.name),  allow_pickle=True)

        # print statistical info
        print('Class frequencies:')
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " % clnt +
                  ', '.join(["%.3f" % np.mean(clnt_y[clnt] == cls) for cls in range(self.n_cls)]) +
                  ', Amount:%d' % clnt_y[clnt].shape[0])
            count += clnt_y[clnt].shape[0]

        print('Total Amount:%d' % count)
        print('--------')

        print("      Test: " +
              ', '.join(["%.3f" % np.mean(tst_y == cls) for cls in range(self.n_cls)]) +
              ', Amount:%d' % tst_y.shape[0])

        return clnt_x, clnt_y, tst_x, tst_y


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        self.train = train
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.X_data = data_x
        self.y_data = data_y
        if not isinstance(data_y, bool):
            self.y_data = data_y.astype('float32')

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        img = self.X_data[idx]
        if self.train:
            img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  # Horizontal flip
            if np.random.rand() > .5:
                # Random cropping
                pad = 4
                extended_img = np.zeros((3, 32 + pad * 2, 32 + pad * 2)).astype(np.float32)
                extended_img[:, pad:-pad, pad:-pad] = img
                dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                img = extended_img[:, dim_1:dim_1 + 32, dim_2:dim_2 + 32]
        img = np.moveaxis(img, 0, -1)
        img = self.transform(img)
        if isinstance(self.y_data, bool):
            return img
        else:
            y = self.y_data[idx]
            return img, y

def split_datasets(dataname, num_clients, num_class, seed, sgm, rule, alpha, data_path='./data', showfig=False):
    """ split datasets <dataname> into <num_clients> parts with the specific split method <rule>
    :param dataname: the name of the dataset, choices in {'mnist', 'emnist', 'CIFAR10', 'CIFAR100', }
    :param num_clients: the quantity of the clients, the datasets will be split into <num_clients> parts
    :param num_class: the number of classes in the dataset
    :param seed: to provide reproducibility
    :param sgm: unbalance parameter,
    :param rule: choices in {'Dirichlet', 'iid'}
    :param alpha: the parameter of Dirichlet distribution, when alpha -> infinity, the distribution becomes iid.
                if rule = 'iid', alpha will be ignored
    :param data_path: the path of predownloaded dataset
    :param showfig: if it is ture, a figure of statistical info on each client will be illstrated
    :return: train_data, train_label. train_data[idx] represents the dataset of the idx-th client
    """
    data_obj = DatasetObject(dataset=dataname,
                             n_client=num_clients,
                             seed=seed,
                             unbalanced_sgm=sgm,
                             rule=rule,
                             rule_arg=alpha,
                             data_path=data_path)
    save_path = "%s_%d_%d_%s_%s" % (dataname, num_clients, seed, rule, alpha)
    print('Split data have been saved in ', save_path)
    if showfig:
        show_statis(data_obj, num_clients, num_class, dataname, save_path)
    return data_obj.clnt_x, data_obj.clnt_y


class DatasetFromDir(data.Dataset):

    def __init__(self, img_root, img_list, label_list, transformer):
        super(DatasetFromDir, self).__init__()
        self.root_dir = img_root
        self.img_list = img_list
        self.label_list = label_list
        self.size = len(self.img_list)
        self.transform = transformer

    def __getitem__(self, index):
        img_name = self.img_list[index % self.size]
        # ********************
        img_path = os.path.join(self.root_dir, img_name)
        img_id = self.label_list[index % self.size]

        img_raw = Image.open(img_path).convert('RGB')
        img = self.transform(img_raw)
        return img, img_id

    def __len__(self):
        return len(self.img_list)

def show_statis(data_obj, num_clients, num_class, dataname, save_path):
    client_statis = []
    for client_id in range(num_clients):
        samples_distribution = [0] * num_class
        data_set, label_set = data_obj.clnt_x[client_id], data_obj.clnt_y[client_id]
        train_loader = torch.utils.data.DataLoader(Dataset(data_set,
                                                           label_set,
                                                           train=True,
                                                           dataset_name=dataname),
                                                   batch_size=1, shuffle=True)

        for _, label in train_loader:
            samples_distribution[int(label.data.numpy())] += 1
        # print(samples_distribution)
        client_statis.append(samples_distribution)
    np.save('./' + save_path + 'statisInfo.npy', client_statis)
    print('Statistical info have been saved in ', save_path)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    client_list = np.array(client_statis)
    client_name_list = []
    for client_id in range(num_clients):
        client_name_list.append('Client' + str(client_id))

    plt.figure(figsize=(10, 10))
    bot = np.zeros([num_class, num_clients])
    for i in range(num_class):
        for j in range(i):
            bot[i][:] += client_list[:, j]
    # print('++++++++++', bot)
    x = range(len(client_name_list))
    for label_id in range(num_class):
        plt.bar(x=x,
                height=client_list[:, label_id],
                width=0.8,
                alpha=0.8,
                color=COLOR_MAP[label_id % len(COLOR_MAP)],
                label=str(label_id),
                bottom=bot[label_id])

    plt.ylabel("Quantity")
    plt.xticks(x, client_name_list)
    plt.xlabel("Client")
    plt.title("Sample distribution")
    plt.legend()
    plt.savefig('./fig_dist_' + str(num_clients) + '.png')
    plt.show()
    plt.close()
