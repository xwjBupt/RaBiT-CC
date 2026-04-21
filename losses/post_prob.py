import torch
from torch.nn import Module

class Post_Prob(Module):
    # 去掉了烦人的 device 参数
    def __init__(self, sigma, c_size, stride, background_ratio, use_background):
        super(Post_Prob, self).__init__()
        assert c_size % stride == 0

        self.sigma = sigma
        self.bg_ratio = background_ratio
        self.use_bg = use_background
        self.softmax = torch.nn.Softmax(dim=0)
        
        # 【修改重点】：生成坐标后，注册为 buffer
        cood = torch.arange(0, c_size, step=stride, dtype=torch.float32) + stride / 2
        cood.unsqueeze_(0)
        self.register_buffer('cood', cood) # Lightning 会自动管理它的设备！

    def forward(self, points, st_sizes):
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)

        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1)
            y = all_points[:, 1].unsqueeze_(1)
            # Tensor Core 会在这里自动发力 (得益于我们在 train.py 开启了 TF32/BF16)
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis = y_dis + x_dis
            dis = dis.view((dis.size(0), -1))

            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            for dis_i, st_size in zip(dis_list, st_sizes):
                if len(dis_i) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis_i, dim=0, keepdim=True)[0], min=0.0)
                        d = st_size * self.bg_ratio
                        bg_dis = (d - torch.sqrt(min_dis))**2
                        dis_i = torch.cat([dis_i, bg_dis], 0)
                    dis_i = -dis_i / (2.0 * self.sigma ** 2)
                    prob = self.softmax(dis_i)
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = [None for _ in range(len(points))]
        return prob_list