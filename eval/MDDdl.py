'''
3 features in 2 timesteps = six values
first value shows first feature in first timestep
second value shows first feature in second timestep
...
'''

import numpy as np
import torch
import torch.nn as nn


# 定义直方图损失函数
class HistoLoss(nn.Module):
    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()

        for i in range(x_real.shape[2]):
            tmp_densities = list()
            tmp_locs = list()
            tmp_deltas = list()
            for t in range(x_real.shape[1]):
                # 取出每个时间步的样本值
                x_ti = x_real[:, t, i].reshape(-1).detach().cpu().numpy()  # 确保是 numpy 数组
                d, b = np.histogram(x_ti, bins=n_bins, density=True)
                tmp_densities.append(torch.tensor(d, dtype=torch.float32))  # 转换为 PyTorch 张量
                delta = b[1:2] - b[:1]
                # delta = b[1] - b[0]  # 计算 bin 宽度
                loc = 0.5 * (b[1:] + b[:-1])  # 计算 bin 中心
                tmp_locs.append(torch.tensor(loc, dtype=torch.float32))  # 转换为 PyTorch 张量
                tmp_deltas.append(torch.tensor(delta, dtype=torch.float32))  # 转换为 PyTorch 张量

            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0).float()

        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1)
                x_ti = x_fake[:, t, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = (relu(self.deltas[i][t] / 2. - dist) > 0).float()
                density = counter.mean(0) / self.deltas[i][t]  # 使用 d 的 bin 宽度
                abs_metric = torch.abs(density - self.densities[i][t])
                loss.append(torch.mean(abs_metric))

        loss_componentwise = torch.stack(loss)
        return loss_componentwise



s1 = torch.tensor([
    [[20, 10, 0], [50, 30, 100]],
    [[50, 30, 100], [180, 20, 20]],
    [[180, 20, 20], [30, 15, 0]]
], dtype=torch.float32)

s2 = torch.tensor([
    [[180, 20, 20], [20, 10, 5]],
    [[20, 10, 5], [30, 16, 1]],
    [[30, 16, 1], [50, 29, 120]]
], dtype=torch.float32)

# s2 = torch.tensor([
#     [[18, 12, 5], [47, 35, 110]],
#     [[47, 35, 110], [200, 18, 40]],
#     [[200, 18, 40], [30, 9, 4]]
# ], dtype=torch.float32)
#
# s2 = torch.tensor([
#     [[20, 10, 20], [180, 20, 100]],
#     [[180, 20, 100], [50, 35, 40]],
#     [[50, 35, 40], [30, 15, 30]]
# ], dtype=torch.float32)


# 实例化 HistoLoss
n_bins = 5  # 设定区间数
histo_loss = HistoLoss(s1, n_bins)

# 计算损失
loss = histo_loss.compute(s2)
print("损失值:", loss)

total_loss = loss.sum()  # 计算所有损失值的总和
print("总损失值:", total_loss)
