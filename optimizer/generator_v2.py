from scipy.stats import special_ortho_group
import torch


class GeneratorV():
    def __init__(self, size):
        self.p_dim = size[0]
        self.k_rank = size[1]
        self.v_list = []
        self.old_v_list = []

        self.initialize()


    def _f(self, v):
        return torch.norm(torch.norm(torch.matmul(self.old_V.T, v)))

    
    def initialize(self):
        V = torch.tensor(special_ortho_group.rvs(self.p_dim)).float()
        self.v_list += [V[:, i].unsqueeze(1) for i in range(V.size(1))]


    def add_v(self):
        V = torch.tensor(special_ortho_group.rvs(self.p_dim)).float()
        new_v = [V[:, i].unsqueeze(1) for i in range(V.size(1))]

        self.v_list += sorted(new_v, key=self._f)

                        
    def get_v(self):
        #if len(self.v_list) < self.k_rank:
        #    self.add_v()

        #while len(self.v_list) < self.k_rank:
        #    self.add_v()

        v_list = []
        
        for _ in range(self.k_rank):
            v = torch.randn((self.p_dim, 1))
            v /= torch.norm(v)
            v_list.append(v)
        V = torch.cat(v_list, dim=1)
        #used_v = self.v_list[:self.k_rank]
        #V = torch.cat(self.v_list[:self.k_rank], dim=1)
        #del self.v_list[:self.k_rank]
        #self.v_list += used_v
        
        return V
