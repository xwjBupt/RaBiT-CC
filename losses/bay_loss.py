from torch.nn.modules import Module
import torch

class Bay_Loss(Module):
    # 同样可以保留 device 参数，或者在 Module 里通过 prob_list 自动推导
    def __init__(self, use_background, device):
        super(Bay_Loss, self).__init__()
        self.device = device
        self.use_bg = use_background

    def forward(self, prob_list, pre_density):
        # 【修改点】：初始化为设备上的 0 张量
        loss = torch.tensor(0.0, device=self.device) 
        
        for idx, prob in enumerate(prob_list): 
            if prob is None:  
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32, device=self.device)
            else:
                N = len(prob)
                target = torch.ones((N,), dtype=torch.float32, device=self.device)
                if self.use_bg:
                    target[-1] = 0.0  
                pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1) 

            # 注意这一行的缩进，确保它在 for 循环内部
            loss += torch.sum(torch.abs(target - pre_count))
            
        loss = loss / len(prob_list)
        return loss