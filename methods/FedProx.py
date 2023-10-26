from methods.local_fn import *


def train_FedProx(data_obj, act_prob, learning_rate, batch_size, epoch,
                  com_amount, print_per, weight_decay,
                  model_func, init_model, sch_step, sch_gamma,
                  save_period, mu, suffix='', trial=True, data_path='',
                  rand_seed=0, lr_decay_per_round=1):
    suffix = 'FedProx_' + suffix
    return train_Fed_common(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                            batch_size=batch_size, epoch=epoch,
                            com_amount=com_amount, print_per=print_per, weight_decay=weight_decay+mu,
                            model_func=model_func, init_model=init_model, sch_step=sch_step, sch_gamma=sch_gamma,
                            save_period=save_period, suffix=suffix, trial=trial, data_path=data_path,
                            rand_seed=rand_seed, lr_decay_per_round=lr_decay_per_round)
