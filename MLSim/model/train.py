import argparse
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model import TAOModel, Config
from dataloader import CustomDataset
from transformer import TAO, EmbedConfig, TransConfig
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--resume', type=str, 
                        help="Model to resume training from",default=None)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Test a model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate For the current Model')
     
    args = parser.parse_args()
    
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    print("Batch sz is %d. Save? %s. Cuda? %s. Dev count: %d. lr: %f" % 
          (args.batch_size,
           str(args.save_model),
           str(use_cuda),
           torch.cuda.device_count(),
           args.lr))
    
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: " ,device)
    

    if args.resume:
        print("resuming")
        model = torch.load(args.resume, map_location=device)
        model.eval()
    else:
        print("starting from scratch")
        embedconfig = EmbedConfig(74+39, 52, 128, 64, 64, 64, 256, 256, 512)
        transconfig = TransConfig(97, 512, 8, 2048, 2, 3)
        model = TAO(embedconfig, transconfig)
    
    print(model, flush=True)
    model.to(device)
    
    # return
    # 使用 DataLoader
    dataFloder = "/mnt/4090/data/shixl/TAO/MLSim/workload/spec2006/bzip2/trainData"
    trainDataNames = ["data.txt0", "data.txt1", "data.txt2", "data.txt3", "data.txt4", "data.txt5", "data.txt6", "data.txt7", "data.txt8"]
    # trainDataNames = ["data.txt9"]
    testDataNames = ["data.txt9"]
    
    trainDataFiles = []
    for dataName in trainDataNames:
        trainDataFiles.append(os.path.join(dataFloder, dataName))
    trainDataSet = CustomDataset(trainDataFiles)
    print(trainDataFiles)
    
    testDataFiles = []
    for dataName in testDataNames:
        testDataFiles.append(os.path.join(dataFloder, dataName))
    testDataSet = CustomDataset(testDataFiles)
    print(testDataFiles, flush=True)
            
    train_data_loader = DataLoader(trainDataSet, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(testDataSet, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # mean = torch.tensor(np.array([5.430755, 323.876932])).to(device)
    # std = torch.tensor(np.array([37.017409, 172.096593])).to(device)
    mean = torch.tensor(np.array([5.430755])).to(device)
    std = torch.tensor(np.array([37.017409])).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40], 0.1)
    
    best, best_idx = [float('inf'), -1]
    
    for epoch in range(1, args.epochs + 1):
        train(args=args, 
              model=model,
              device=device, 
              train_loader=train_data_loader,
              optimizer=optimizer,
              epoch=epoch)
        # scheduler.step()
        
        multiplier=1
        if (epoch % multiplier == 0):
                t = test(args=args, 
                       model=model, 
                       device=device,
                       test_loader=test_data_loader,
                       mean=mean,
                       std=std)
                if  t < best:
                    print("Got improvement at {} (from {:.4f} to {:.4f}".format(epoch,best,t))
                    best , best_idx = (t,epoch)
                    if (args.save_model) :
                        if use_cuda:
                            torch.save(model, "checkpoint/model-update1-wd/best.pt" )
                        else:
                            torch.save(model, "best.pt" )
                    
        # if epoch > best_idx + 3*multiplier : #e.g. best is at 100, we exit at 401
        #     print("Ending at {} as there's been no improvement in 300 epochs. Best was at {}".format(epoch,best_idx))
        #     break

        if (args.save_model) and (epoch % 10 == 0):
            print("Saving")
            if use_cuda:
                torch.save(model,
                           "checkpoint/model-update1-wd/%d.pt" % epoch)
            else:
                torch.save(model,
                           "checkpoint/model-update1-wd/%d.pt" % epoch)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train() 
    print("Device:" ,device, flush=True)
    # loss = nn.MSELoss() 
    # loss = nn.L1Loss() #(reduction=None)
    train_weight = [0.5714338,  0.88887177, 8.00036268]
    train_weight = torch.FloatTensor(train_weight).cuda()
    loss = nn.CrossEntropyLoss(train_weight)
    
    start_time = time.time()
    cur_time = start_time
    loss_sum = 0
    minibatches = 0
    #print("Train loader has %d minibatches." % len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # print("Forwarding model")
        # print("data shape:", data.shape)
        
        output = model(data)
        value = loss(output,target)
        # value = weighted_l1_loss(output, target, weight)
        
        optimizer.zero_grad()
        value.backward()
        loss_sum += value.item()
        minibatches += 1

        optimizer.step()
        prev_time = cur_time
        cur_time = time.time()
        if batch_idx%20 == 0:
            print("Epoch %d Minibatch %d took %f " \
                "Epoch of %d mbs would be %f "\
                "(Avg so far is %f)"\
                "loss is %f" % (epoch,
                                        batch_idx,
                                        cur_time - prev_time,
                                        len(train_loader),
                                        (cur_time - prev_time)*len(train_loader), 
                                        (cur_time-start_time)/(batch_idx + 1),
                                        value.item()), flush=True)  
        
    print("Epoch %d  time was %f" % (epoch, time.time() - start_time), flush=True)
    print("Epoch %d  avg tring loss was %f" % (epoch, loss_sum/minibatches))

def test(args, model, device, test_loader,mean,std):
    model.eval()
    
    test_weight = [0.5714289,  0.88888889, 7.999936  ]
    class_avg = [0, 1, 166.0841433268534]
    test_weight = torch.FloatTensor(test_weight).cuda()
    loss = nn.CrossEntropyLoss(test_weight)
    
    total_acc = 0.0
    num = 0
    std_loss = 0.0
    iteration = 0
    target_fetch = 0.0
    pred_fetch = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = torch.max(output, dim=1)[1]
            test_acc = evaluate(pred.cpu().numpy(), target.cpu().numpy(), 'sum')
            total_acc += test_acc
            num += output.shape[0]
            std_loss += loss(output, target)
            iteration += 1
            
            target_fetch += calfetch(target.cpu().numpy(), class_avg)
            pred_fetch += calfetch(pred.cpu().numpy(), class_avg)
            
    std_acc = total_acc/num*100
    std_loss /= iteration
    diff_fetch = target_fetch - pred_fetch
    print('>>>>Test set avg train loss: {:.4f}\n'
          '>>>>Test set accuracy {:.2f}%\n'
          '>>>>>overall fetch target {:.4f} overall fetch predict {:.4f}\n'
          '>>>>>overall fetch diff {:.4f} '.format(
              std_loss,
              std_acc,
              target_fetch, pred_fetch,
              diff_fetch), 
          flush=True)
    return std_acc.item()

if __name__ == '__main__':
    print("Now time is", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), "\n")
    main()
    