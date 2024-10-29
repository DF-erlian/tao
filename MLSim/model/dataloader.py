import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 自定义 Dataset 类
class CustomDataset(Dataset):
    def __init__(self, data_file):
        # 读取数据
        self.data = np.loadtxt(data_file)
        self.inputs = torch.tensor(self.data[:, 2:], dtype=torch.float32)  # 输入特征
        self.outputs = torch.tensor(self.data[:, :2], dtype=torch.float32)  # 输出值
        self.robsize = 96  # 每个批次的大小
    def __len__(self):
        # 返回数据集的大小
        return len(self.data) - self.robsize

    def __getitem__(self, idx):
        # 返回指定索引的数据项
        return self.inputs[idx:idx+self.robsize, : ], self.outputs[idx+self.robsize-1, :]

# 使用 DataLoader
data_file = '/worksapce/TAO/MLSim/workload/spec2006/bzip2/trainData.txt0'  # 数据文件路径
dataset = CustomDataset(data_file)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)

# 遍历 DataLoader
for inputs, outputs in dataloader:
    print("Inputs:", inputs)
    print("Inputs shape:", inputs.shape)
    print("Outputs:", outputs)
    break
    # 这里可以添加模型训练或验证的代码
# 

