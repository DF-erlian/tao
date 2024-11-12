import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader

fetch_mean = 5.430755
fetch_std = 37.01740

exec_mean = 323.876932
exec_std = 172.09659

# 自定义 Dataset 类
class CustomDataset(Dataset):
    def __init__(self, dataFiles):
        # 读取数据
        self.windows = 96
        self.data = []
        self.dataIndex = []
        for dataFile in dataFiles:
            data = np.loadtxt(dataFile)
            mean = np.mean(data[:, 2:], axis=0) # 标准化, 训练、测试集仅一个文件
            std = np.std(data[:, 2:], axis = 0)
            std[std==0] = 1
            data[:,2:] = (data[:,2:] - mean)/std
            self.data.append(data)
            self.dataIndex.append(len(data) - self.windows)
        
    def __len__(self):
        # 返回数据集的大小
        sum = 0
        for index in self.dataIndex:
            sum += (index)
        return sum

    def __getitem__(self, idx):
        # 返回指定索引的数据项
        for k, dataIdx in enumerate(self.dataIndex):
            if(idx > dataIdx):
                idx -= dataIdx
                continue
            else :
                input = self.data[k][idx:idx+self.windows, 2:]
                output = self.data[k][idx+self.windows-1, 0:1]
        # output[0] = (output[0])/fetch_std # 标准化, 训练、测试集仅一个文件
        # output[1] = (output[1] - exec_mean)/exec_std
        return torch.tensor(input, dtype=torch.float32), torch.tensor(output, dtype=torch.float32)

# dataFloder = "/mnt/mix/ssd/shixl/TAO/MLSim/workload/spec2006/bzip2/trainData"
# trainDataNames = ["data.txt0", "data.txt1"]

# trainDataFiles = []
# for dataName in trainDataNames:
#     trainDataFiles.append(os.path.join(dataFloder, dataName))
# trainDataSet = CustomDataset(trainDataFiles)

# train_data_loader = DataLoader(trainDataSet, batch_size=100, shuffle=True, num_workers=2)

# for inputs, outputs in train_data_loader:
#     print("Inputs:", inputs)
#     print("Inputs shape:", inputs.shape)
#     print("Outputs:", outputs)
#     break
