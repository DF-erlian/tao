import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import convolve1d

# fetch_mean = 5.430755
# fetch_std = 37.01740

# exec_mean = 323.876932
# exec_std = 172.09659

# 自定义 Dataset 类
class CustomDataset(Dataset):
    def __init__(self, dataFiles):
        # 读取数据
        self.windows = 96+1
        self.data = None
        self.dataIndex = []
        for dataFile in dataFiles:
            data = np.loadtxt(dataFile)
            if self.data is None:
                self.data = data
            else:
                self.data = np.vstack((self.data, data))
        
        mean = np.mean(self.data[:, 2:], axis=0)
        std = np.std(self.data[:, 2:], axis=0)
        std[std==0] = 1
        self.data[:, 2:] = (self.data[:,2:] - mean)/std
        
        self.modified_labels = self._prepare_weights()
         
        
    def __len__(self):
        return self.data.shape[0] - self.windows
    
    def __getitem__(self, idx):
        input = self.data[idx:idx+self.windows, 2:]
        output = self.modified_labels[idx+self.windows-1]
        # weight = np.asarray([self.weights[idx+self.windows-1]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])
        
        return torch.tensor(input, dtype=torch.float32), output

    def _prepare_weights(self):
        labels = self.data[:, 0].astype(int).tolist()
        modified_labels = [label if label in [0, 1] else 2 for label in labels]
        
        sum_labels = [0, 0, 0]
        for label in labels:
            if label == 0:
                sum_labels[0] += label
            elif label == 1:
                sum_labels[1] += label
            else:
                sum_labels[2] += label
        
        # print(len(modified_labels))
        class_counts = np.bincount(modified_labels)
        print(class_counts)
        total_samples = len(labels)
        
        weights = total_samples / (len(class_counts) * class_counts)
        print(weights)
        
        avg_labels = [sum/count for (sum, count) in zip(sum_labels, class_counts)]
        print(avg_labels)
        
        return modified_labels
    
# dataFloder = "/mnt/mix/ssd/shixl/TAO/MLSim/workload/spec2006/bzip2/trainData"
# trainDataNames = ["data.txt0", "data.txt1", "data.txt2", "data.txt3", "data.txt4", "data.txt5", "data.txt6", "data.txt7", "data.txt8"]
# # trainDataNames = ["data.txt9"]

# trainDataFiles = []
# for dataName in trainDataNames:
#     trainDataFiles.append(os.path.join(dataFloder, dataName))
# trainDataSet = CustomDataset(trainDataFiles)

# train_data_loader = DataLoader(trainDataSet, batch_size=1000, shuffle=True, num_workers=2)

# for inputs, outputs in train_data_loader:
#     # print("Inputs:", inputs)
#     print("Inputs shape:", inputs.shape)
#     print("Outputs shape:", outputs.shape)
#     break
