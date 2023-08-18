import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
import time
import random
import math
from scipy.stats import special_ortho_group

from .generator_v import *
from .util import *

class CEclOptimizer(Optimizer):
    def __init__(self, params, node_id: int, adj_node_ids: list, lr=1e-5, itr_per_round=5, comp_rate=0.2, theta=0.5, device="cuda"):
        """
        Parameter
        -------------
        params : torch.Parameter
            mode parameters

        node_id : int
            node id

        adj_node_ids : list of int
            list of node id of neighbors
        
        lr : float
            learning rate
        
        itr_per_round : int
            K in the paper
        
        decice : string
            "cuda:*" or "cpu"
        """
        self.lr = lr
        self.theta = theta
        self.node_id = node_id
        self.adj_node_ids = adj_node_ids
        self.device = device
        self.counter = dict.fromkeys(self.adj_node_ids, 0) # This is used for when to transmit parameters.

        self.counter_z = dict.fromkeys(self.adj_node_ids, True) # This is used for checking which Z (z_even or z_odd) is used.

        self.itr_per_round = itr_per_round
        self.comp_rate = comp_rate # compression rate
        self.l2_penalty = 0.001
        
        defaults = dict(lr=lr)
        super(CEclOptimizer, self).__init__(params, defaults)


        self.n_sent_params = 0 # the number of parameters to be sent.

                        
        # generate initial dual variables.
        for group in self.param_groups:
            group["dual_z"] = []
            group["dual_y"] = []
            group["adj_params"] = []
            group["generator_v"] = []
            
            for p in group["params"]:
                dual_z = {}
                dual_y = {}
                
                adj_params = {}
                generator_v = {}
                adj_generator_v = {}
                
                for adj_node_id in adj_node_ids:
                    dual_z[adj_node_id] = torch.zeros_like(p, device=device)
                    dual_y[adj_node_id] = torch.zeros_like(p, device=device)

                    adj_params[adj_node_id] = torch.zeros_like(p, device=device)
                    generator_v[adj_node_id] = GeneratorMask(p.size(), self.comp_rate, self.device)
                    
                group["dual_z"].append(dual_z)
                group["dual_y"].append(dual_y)
                group["adj_params"].append(adj_params)
                group["generator_v"].append(generator_v)

                
        # generate A_{i|j}
        self.state["A"] = {}
        for adj_node_id in adj_node_ids:
            if self.node_id < adj_node_id:
                self.state["A"][adj_node_id] = 1.0
            else:
                self.state["A"][adj_node_id] = -1.0
        
        
    @torch.no_grad()
    def initialize(self):
        """
        exchange seed values for generator.
        """
        pass

    """
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        
        for group in self.param_groups:
            lr = self.lr #group['lr']
            mu = 1. / lr 
            
            for p, dual_z, dual_y, in zip(group['params'], group["dual_z"], group["dual_y"]):

                p_dim = p.size(0)
                alpha = 1.0 / (lr * self.itr_per_round/self.comp_rate * len(self.adj_node_ids))

                # update model parameters W_i
                tmp = torch.zeros_like(p)
                for adj_node_id in self.adj_node_ids:
                    tmp += p.data - self.state["A"][adj_node_id]*dual_z[adj_node_id].data

                p.data = p.data - lr* (p.grad.data + self.l2_penalty*p.data + alpha*(tmp - lr*len(self.adj_node_ids)*p.grad.data))
                
                # update dual variables Y_{i, j}
                for adj_node_id in self.adj_node_ids:
                    dual_y[adj_node_id] = dual_z[adj_node_id] - 2. * self.state["A"][adj_node_id] * p.data

                    
        if closure is not None:
            loss = closure()

        self.update(self.adj_node_ids)
            
        return loss
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        
        for group in self.param_groups:
            lr = self.lr #group['lr']
            alpha = 1.0 / (lr * (self.itr_per_round/self.comp_rate -1) * len(self.adj_node_ids))
            mu = 1. / lr    

            
            for p, dual_z, dual_y in zip(group['params'], group["dual_z"], group["dual_y"]):
                
                # update model parameters W_i
                tmp = torch.zeros_like(p)
                for adj_node_id in self.adj_node_ids:
                    tmp += self.state["A"][adj_node_id]*dual_z[adj_node_id].data

                p.data = (mu*p.data - p.grad.data - self.l2_penalty*p.data + alpha*tmp) / (mu + alpha*len(self.adj_node_ids))
                
                # update dual variables Y_{i, j}
                for adj_node_id in self.adj_node_ids:
                    dual_y[adj_node_id] = dual_z[adj_node_id] - 2. * self.state["A"][adj_node_id] * p.data
            
        if closure is not None:
            loss = closure()

        self.update(self.adj_node_ids)
            
        return loss

        
    @torch.no_grad()
    def update(self, node_ids):
        """
        Parameters
        ----------
        node_ids : list of int
        
        ----------

        If self.counter[node_id]%self/itr_per_round, transmit parameters.
        """
        exchanged_node_ids = []
        
        for node_id in node_ids:
            if self.node_id < node_id:
                if self.counter[node_id]%self.itr_per_round == 0:
                    exchanged_node_ids.append(node_id)
                    
                    dist.send(tensor=torch.tensor(1.0), dst=node_id, tag=111)
                    
                    #self.send_param(node_id)
                    #self.recv_param(node_id)

                    self.send_dual(node_id)
                    self.recv_dual(node_id)

                else:
                    dist.send(tensor=torch.tensor(0.0), dst=node_id, tag=111)
            else:
                flag = torch.tensor(-1.0)
                dist.recv(tensor=flag, src=node_id, tag=111)

                if flag.item() == 1.0:
                    exchanged_node_ids.append(node_id)

                    #self.recv_param(node_id)
                    #self.send_param(node_id)
                    
                    self.recv_dual(node_id)
                    self.send_dual(node_id)

                else:
                    pass
            
            self.counter[node_id] += 1

    """
    @torch.no_grad()
    def send_param(self, node_id):
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                dist.send(tensor=p.data.to("cpu"), dst=node_id, tag=i)

                
    @torch.no_grad()
    def recv_param(self, node_id):
        for group in self.param_groups:        
            for i, adj_p in enumerate(group["adj_params"]):    
                tmp = torch.zeros_like(adj_p[node_id], device="cpu")
                dist.recv(tensor=tmp, src=node_id, tag=i)
                adj_p[node_id].data = tmp.data.to(self.device)        
    """
    
    @torch.no_grad()
    def send_dual(self, node_id):
        for group in self.param_groups:
            for i, (dual_y, gen) in enumerate(zip(group["dual_y"], group["generator_v"])):

                gen[node_id].change_comp_rate(self.comp_rate)
                mask = gen[node_id].get_v()
                
                n_param = dist_send_sparse((mask*dual_y[node_id]), dst=node_id, tag=i)

                self.n_sent_params += n_param
                
                # if we use the following code, we can transmit full-precision y.
                #dist.send(tensor=dual_y[node_id].data.to("cpu"), dst=node_id, tag=i)

                
    @torch.no_grad()
    def recv_dual(self, node_id):
        for group in self.param_groups:        
            #for i, (dual_y, dual_z) in enumerate(zip(group["dual_y"], group["dual_z"])):
            for i, dual_z in enumerate(group["dual_z"]):
                
                """
                adj_dual_y = dist_recv_sparse(dual_z[node_id].size(), src=node_id, tag=i).to(self.device)
                                
                gen[node_id].change_comp_rate(self.comp_rate)
                mask = gen[node_id].get_v().to(self.device)

                dual_z[node_id] = dual_z[node_id] + self.theta*(adj_dual_y - mask*dual_z[node_id])
                """

                diff = recv_and_compute_diff(dual_z[node_id], src=node_id, tag=i)
                dual_z[node_id] = dual_z[node_id] + self.theta*diff
                
                ## if we use the following code, we can recieve full-precision y.
                # adj_dual_y = torch.zeros_like(dual_z[node_id], device="cpu")
                # dist.recv(tensor=adj_dual_y, src=node_id, tag=i)
                # adj_dual_y = adj_dual_y.to(self.device)
                # gen[node_id].change_comp_rate(self.comp_rate)
                # mask = gen[node_id].get_v().to(self.device)
                # dual_z[node_id] = dual_z[node_id] + mask*(self.theta*(adj_dual_y - dual_z[node_id]))
                
                
    @torch.no_grad()
    def param_diff(self):
        diff = 0.
        for group in self.param_groups:
            for i, (p, adj_p) in enumerate(zip(group["params"], group["adj_params"])):
                for node_id in self.adj_node_ids:
                    diff += torch.norm(p - adj_p[node_id]).detach().cpu()
        #return diff
        return -1
