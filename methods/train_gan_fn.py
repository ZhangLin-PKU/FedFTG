import numpy as np
import torch
import torch.nn as nn

def get_class_number(data_obj):
    clnt_cls_num = np.zeros((data_obj.n_client, data_obj.n_cls))
    clnt_y = data_obj.clnt_y
    for i in range(data_obj.n_client):
        for j in clnt_y[i]:
            clnt_cls_num[i, j] += 1
    return clnt_cls_num

def generate_labels(number, cls_num):
    labels = np.arange(number)
    proportions = cls_num / cls_num.sum()
    proportions = (np.cumsum(proportions) * number).astype(int)[:-1]
    labels_split = np.split(labels, proportions)
    for i in range(len(labels_split)):
        labels_split[i].fill(i)
    labels = np.concatenate(labels_split)
    np.random.shuffle(labels)
    return labels.astype(int)

def get_batch_weight(labels, cls_clnt_weight):
    bs = labels.size
    num_clients = cls_clnt_weight.shape[1]
    batch_weight = np.zeros((bs, num_clients))
    batch_weight[np.arange(bs), :] = cls_clnt_weight[labels, :]
    return batch_weight

def compute_backward_flow_G_dis(z, y_onehot, labels,
                                generator, student, teacher,
                                weight, num_clients, train_fedgen_feature=False):
    lambda_cls = 1.0
    lambda_dis = 1.0
    # cls_criterion = nn.CrossEntropyLoss().cuda()
    cls_criterion = nn.CrossEntropyLoss(reduction='mean').cuda()
    diversity_criterion = DiversityLoss(metric='l1').cuda()

    y = torch.Tensor(labels).long().cuda()

    fake = generator(z, y_onehot)
    t_logit = teacher(fake, train_fedgen_feature)
    s_logit = student(fake, train_fedgen_feature)
    loss_md = - torch.mean(
        torch.mean(torch.abs(s_logit - t_logit.detach()), dim=1) * weight)

    # loss_cls = cls_criterion(t_logit, y)
    loss_cls = torch.mean(cls_criterion(t_logit, y) * weight.squeeze())

    loss_ap = diversity_criterion(z.view(z.shape[0],-1), fake)
    loss = loss_md + lambda_cls * loss_cls + lambda_dis * loss_ap / num_clients
    loss.backward()
    return loss, loss_md, loss_cls, loss_ap

class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))
