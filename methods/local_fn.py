import copy
import torch
from torch.utils import data
from utils.util_dataset import Dataset
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

# Global parameters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_norm = 10


# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, batch_size, w_decay=None):
    acc_overall = 0
    loss_overall = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # batch_size = min(6000, data_x.shape[0])
    batch_size = min(2000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_overall += loss.item()

            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_tst
    if w_decay is not None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay / 2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst


# --- Helper functions

def avg_models(mdl, clnt_models, weight_list):
    n_node = len(clnt_models)
    dict_list = list(range(n_node))
    for i in range(n_node):
        dict_list[i] = copy.deepcopy(dict(clnt_models[i].state_dict()))

    param_0 = clnt_models[0].state_dict()

    for name, param in param_0:
        param_ = weight_list[0] * param.data
        for i in list(range(1, n_node)):
            param_ = param_ + weight_list[i] * dict_list[i][name].data
        dict_list[0][name].data.copy_(param_)

    mdl.load_state_dict(dict_list[0])

    # Remove dict_list from memory
    del dict_list

    return mdl


def train_model(model, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch, print_per, weight_decay,
                dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, acc_tst, loss_tst, optimizer.param_groups[0]['lr']))
    else:
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, optimizer.param_groups[0]['lr']))

    model.train()

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            if print_test:
                loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
                      % (e + 1, acc_trn, loss_trn, acc_tst, loss_tst, optimizer.param_groups[0]['lr']))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
                    e + 1, acc_trn, loss_trn, optimizer.param_groups[0]['lr']))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

def train_model_moon(model, old_models, glb_model, trn_x, trn_y, tst_x, tst_y, mu, tau, learning_rate, batch_size, epoch, print_per, weight_decay,
                dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    cos_dis = torch.nn.CosineSimilarity(dim=-1)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, acc_tst, loss_tst, optimizer.param_groups[0]['lr']))
    else:
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, optimizer.param_groups[0]['lr']))

    model.train()

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            embed_cur, y_pred = model(batch_x, flag_2=True)

            embed_glb, _ = glb_model(batch_x, flag_2=True)
            dis_pos = cos_dis(embed_cur, embed_glb)
            dis_logits = dis_pos.reshape(-1,1)
            for old_net in old_models:
                # old_net.to(device)
                embed_old, _ = old_net(batch_x, flag_2=True)
                dis_neg = cos_dis(embed_cur, embed_old)
                dis_logits = torch.cat((dis_logits, dis_neg.reshape(-1,1)), dim=1)
                # old_net.to('cpu')

            dis_logits /= tau
            labels = torch.zeros(batch_x.size(0)).cuda().long()

            loss_cls = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_con = mu * loss_fn(dis_logits, labels)

            loss = (loss_cls + loss_con) / list(batch_y.size())[0]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            if print_test:
                loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                print(
                    "Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Loss_cls: %.4f, Loss_con: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
                    % (e + 1, acc_trn, loss.item(), loss_cls.item(), loss_con.item(), acc_tst, loss_tst, optimizer.param_groups[0]['lr']))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Loss_cls: %.4f, Loss_con: %.4f, LR: %.4f" % (
                    e + 1, acc_trn, loss.item(), loss_cls.item(), loss_con.item(), optimizer.param_groups[0]['lr']))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

def train_scaffold_mdl(model, model_func, state_params_diff, trn_x, trn_y,
                       learning_rate, batch_size, n_minibatch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    n_iter_per_epoch = int(np.ceil(n_trn / batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)

    count_step = 0
    is_done = False

    step_loss = 0
    n_data_step = 0
    for e in range(epoch):
        # Training
        ###
        if is_done:
            break
        ###

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            ####
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            #
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            # Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            # print(local_par_list.size())
            # print(state_params_diff.size())
            loss_algo = torch.sum(local_par_list * state_params_diff)

            loss = loss_f_i + loss_algo

            ###
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0]
            n_data_step += list(batch_y.size())[0]

            if count_step % print_per == 0:
                step_loss /= n_data_step
                if weight_decay is not None:
                    # Add L2 loss to complete f_i
                    params = get_mdl_params([model], n_par)
                    step_loss += weight_decay / 2 * np.sum(params * params)

                print("Step %3d, Training Loss: %.4f, LR: %.5f"
                      % (count_step, step_loss, optimizer.param_groups[0]['lr']))
                step_loss = 0
                n_data_step = 0

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.state_dict()))
    idx = 0
    for name, param in dict(mdl.state_dict()).items():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(model_list, n_par=None):
    if n_par is None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in dict(exp_mdl.state_dict()).items():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in dict(mdl.state_dict()).items():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

def get_mdl_nonbn_idx(model_list, name_par=None):
    if name_par is None:
        exp_mdl = model_list[0]
        name_par = []
        for name, param in exp_mdl.named_parameters():
            name_par.append(name)

    idx_list = [[]] * len(model_list)
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in dict(mdl.state_dict()).items():
            temp = param.data.cpu().numpy().reshape(-1)
            if name in name_par:
                idx_list[i].extend(list(range(idx, idx + len(temp))))
            idx += len(temp)
    # print(len(idx_list[0]))
    return np.array(idx_list)

def get_mdl_bn_idx(model_list, name_par=None):
    if name_par is None:
        exp_mdl = model_list[0]
        name_par = []
        for name, param in exp_mdl.named_parameters():
            name_par.append(name)

    idx_list = [[]] * len(model_list)
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in dict(mdl.state_dict()).items():
            temp = param.data.cpu().numpy().reshape(-1)
            if name not in name_par:
                idx_list[i].extend(list(range(idx, idx + len(temp))))
            idx += len(temp)
    # print(len(idx_list[0]))
    return np.array(idx_list)

def train_model_alg(model, model_func, alpha_coef, avg_mdl_param, hist_params_diff, trn_x, trn_y,
                    learning_rate, batch_size, epoch, print_per,
                    weight_decay, dataset_name, sch_step, sch_gamma):
    n_trn = trn_x.shape[0]

    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef + weight_decay)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            # Get f_i estimate
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_f_i = loss_f_i / list(batch_y.size())[0]

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
            # for name, param in dict(model.state_dict()).items():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))

            loss = loss_f_i + loss_algo

            ###
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e + 1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay is not None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef + weight_decay) / 2 * np.sum(params * params)

            print("Epoch %3d, Training Loss: %.4f, LR: %.5f"
                  % (e + 1, epoch_loss, optimizer.param_groups[0]['lr']))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model


def train_Fed_common(data_obj, act_prob, learning_rate, batch_size, epoch,
                     com_amount, print_per, weight_decay,
                     model_func, init_model, sch_step, sch_gamma,
                     save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    # suffix = 'Fed_' + suffix
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
        save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_client

    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    fed_mdls_all = list(range(n_save_instances))

    trn_perf_sel = np.zeros((com_amount, 2))
    trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2))
    tst_perf_all = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%s/%s' % (data_path, data_obj.name, suffix))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr // save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr // save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    trn_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_perf_all[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_trn_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    tst_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_perf_all[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if trial or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        # clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

            all_model = model_func().to(device)
            all_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            # Fix randomness
            inc_seed = 0
            while True:
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            # del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False

                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.state_dict())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_models[clnt] = train_model(clnt_models[clnt], trn_x, trn_y,
                                                tst_x, tst_y,
                                                learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per,
                                                weight_decay,
                                                data_obj.dataset, sch_step, sch_gamma)

                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]

            # Scale with weights

            avg_model = set_client_from_params(model_func(), np.sum(
                clnt_params_list[selected_clnts] * weight_list[selected_clnts] / np.sum(weight_list[selected_clnts]),
                axis=0))
            all_model = set_client_from_params(model_func(),
                                               np.sum(clnt_params_list * weight_list / np.sum(weight_list), axis=0))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             avg_model, data_obj.dataset, 0)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            ###
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                             all_model, data_obj.dataset, 0)
            tst_perf_all[i] = [loss_tst, acc_tst]

            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             all_model, data_obj.dataset, 0)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            writer.add_scalars('Loss/train_wd',
                               {
                                   'Sel clients':
                                       get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset, weight_decay)[0],
                                   'All clients':
                                       get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset, weight_decay)[0]
                               }, i
                               )

            writer.add_scalars('Loss/train',
                               {
                                   'Sel clients': trn_perf_sel[i][0],
                                   'All clients': trn_perf_all[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/train',
                               {
                                   'Sel clients': trn_perf_sel[i][1],
                                   'All clients': trn_perf_all[i][1]
                               }, i
                               )

            writer.add_scalars('Loss/test',
                               {
                                   'Sel clients': tst_perf_sel[i][0],
                                   'All clients': tst_perf_all[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Sel clients': tst_perf_sel[i][1],
                                   'All clients': tst_perf_all[i][1]
                               }, i
                               )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(all_model.state_dict(), '%sModel/%s/%s/%dcom_all.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_sel[:i + 1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_sel[:i + 1])

                np.save('%sModel/%s/%s/%dcom_trn_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_all[:i + 1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_all[:i + 1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%dcom_trn_perf_all.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_all.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))


            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model
                fed_mdls_all[i // save_period] = all_model

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all
