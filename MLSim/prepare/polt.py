import matplotlib.pyplot as plt
import numpy as np

x_values_martix = []
y_values_martix = []


with open('log.txt', 'r') as file:
    for line_number, line in enumerate(file):
        if(line_number < 2):
            continue
        parts = line.split()
        number = int(parts[0])
        frequency = int(parts[2])
        x_values_martix.append(number)
        y_values_martix.append(frequency)

x_values_bzip = []
y_values_bzip = []

with open('log_bzip.txt', 'r') as file:
    for line_number, line in enumerate(file):
        if(line_number < 2):
            continue
        parts = line.split()
        number = int(parts[0])
        frequency = int(parts[2])
        x_values_bzip.append(number)
        y_values_bzip.append(frequency)
        
        
plt.figure(figsize=(200, 100))        

plt.plot(x_values_martix, y_values_martix, color='red', linestyle='-', label='martix')
plt.plot(x_values_bzip, y_values_bzip,  color='blue', linestyle='-', label='bzip')

# tick_values = np.arange(min(min(x_values_martix), min(x_values_bzip)), 
#                         max(max(x_values_martix), max(x_values_bzip)) + 1, 
#                         step=10)
# plt.xticks(tick_values)

plt.title('Frequency of Occurrences', fontsize=40)
plt.xlabel('Number', fontsize=32)
plt.ylabel('Frequency', fontsize=32)
plt.grid()
plt.legend(fontsize=28)  
plt.tick_params(axis='both', labelsize=12)
plt.savefig("plot.jpg", dpi=300)      