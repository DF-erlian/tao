import numpy as np
import matplotlib.pyplot as plt
import os

dataFloder = "/mnt/mix/ssd/shixl/TAO/MLSim/workload/spec2006/bzip2/trainData"
# dataNames = ["data.txt0", "data.txt1", "data.txt2", "data.txt3", "data.txt4", "data.txt5", "data.txt6", "data.txt7", "data.txt8", "data.txt9"]
dataNames = ["data.txt9"]

std = []
mean = []

fetch_var = []
exec_var = []

for dataName in dataNames:
    dataPath = os.path.join(dataFloder, dataName)
    data = np.loadtxt(dataPath)
    print(data.shape)
    
    fetch_var.extend(data[:,0])
    exec_var.extend(data[:,1])
    
    # break
fetch_var = np.array(fetch_var).reshape(-1)
# print(fetch_var.sum())

print(fetch_var.shape)
print(fetch_var.max(), fetch_var.min())
print("fetch mean: ", fetch_var.mean())
print("fetch std : ", fetch_var.std())

plt.hist(fetch_var, bins=2500, color='skyblue', edgecolor='black')
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.ylim(0,500)
plt.savefig("distribution.jpg")

exec_var = np.array(exec_var)
print(exec_var.shape)
print(exec_var.max(), exec_var.min())
print("exec mean: ", exec_var.mean())
print("exec std :", exec_var.std())