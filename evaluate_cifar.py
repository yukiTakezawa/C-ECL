import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from collections import defaultdict, OrderedDict
import pickle
from tqdm import tqdm
import random
import math
import argparse    
import json
import random
from torch.optim.lr_scheduler import LambdaLR

from model.lenet_cifar10 import *
from optimizer.cecl_optimizer import *

from data.loader import *


def run(rank, size, datasets, config):
    # initialize the model parameters with same seed value.
    torch.manual_seed(0)

    torch.set_num_threads(1)

    net = LeNetCifar10(device=config["device"][rank]).to(config["device"][rank])

    net.to(config["device"][rank])
    
    loaders = datasets_to_loaders(datasets, config["batch"])


    if config["method"] == "cecl":
        optimizer = CEclOptimizer(params=net.parameters(), node_id=rank, adj_node_ids=config["nw"][rank], lr=config["lr"], itr_per_round=config["itr_per_round"], comp_rate=config["comp_rate"], device=config["device"][rank], theta=config["theta"])
        optimizer.initialize()

    
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "diff_param": []}
    history["all_train_loss"]  = []
    history["all_train_acc"] = []
    history["n_sent_params"] = []

    count_epoch = 0
    
    with tqdm(range(config["epochs"]), desc=("node "+str(rank)), position=rank) as pbar:
        for epoch in pbar:

            if config["method"] == "cecl":
                if count_epoch<1:
                    optimizer.comp_rate = 1.0
                else:
                    optimizer.comp_rate = config["comp_rate"]
            
                    
            train_loss, train_acc = net.run(loaders, optimizer)

            if count_epoch % 10 ==0:
                all_train_loss, all_train_acc = net.run_all_train(loaders)
                test_loss, test_acc = net.run_test(loaders)
                            
                # save loss and accuracy
                history["train_loss"] += [train_loss]
                history["test_loss"] += [test_loss]
                history["train_acc"] += [train_acc]
                history["test_acc"] += [test_acc]
                
                history["all_train_loss"] += [all_train_loss]
                history["all_train_acc"] += [all_train_acc]
                history["diff_param"].append(optimizer.param_diff())
            
                pbar.set_postfix(OrderedDict(loss=(round(train_loss, 2), round(test_loss, 2)), acc=(round(train_acc, 2), round(test_acc, 2)), diff=(history["diff_param"][-1])))
            count_epoch += 1
            
    history["n_sent_params"] = optimizer.n_sent_params
    pickle.dump(history, open(config["log_path"] + "node" + str(rank) + ".pk", "wb"))
    
    
def init_process(rank, size, datasets, config, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = config["port"] #'29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, datasets, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PowerECL')
    parser.add_argument('method', default="powerecl", type=str)    
    parser.add_argument('log', default="./log/powerecl", type=str)
    parser.add_argument('--port', default='29500', type=str)
    parser.add_argument('--nw', default="config/ring3_iid.json", type=str)
    parser.add_argument('--alpha', default=10.0, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--itr_per_round', default=5, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--cuda', default=None, type=str) # if None, use "nw" file parameter.
    parser.add_argument('--power_itr', default=10, type=int)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument("--theta", default=0.5, type=float)
    parser.add_argument("--comp_rate", default=0.2, type=float)
    args = parser.parse_args()

    config = defaultdict(dict)
    config["lr"] = args.lr
    config["alpha"] = args.alpha
    config["epochs"] = args.epoch
    config["log_path"] = args.log
    config["method"] = args.method
    config["port"] = args.port
    config["itr_per_round"] = args.itr_per_round
    config["batch"] = 100
    config["power_itr"] = args.power_itr
    config["theta"] = args.theta
    config["comp_rate"] = args.comp_rate
    
    config_json = json.load(open(args.nw, "r"))
    
    n_node = len(config_json)
    
    config["nw"] = [config_json["node" + str(i)]["adj"] for i in range(n_node)]
    config["node_label"] = [config_json["node" + str(i)]["n_class"] for i in range(n_node)]

    if args.cuda is None:
        config["device"] = [config_json["node" + str(i)]["cuda"] for i in range(n_node)]
    else:
        config["device"] = [args.cuda for _ in range(n_node)]

    datasets = load_CIFAR10_hetero(config["node_label"], batch=config["batch"], val_rate=0.0)
    
    processes = []
    mp.set_start_method("spawn")
    for rank in range(n_node):
        node_datasets = {"train": datasets["train"][rank], "val": datasets["val"], "all_train": datasets["all_train"], "test": datasets["test"]}
        p = mp.Process(target=init_process, args=(rank, n_node, node_datasets, config, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
