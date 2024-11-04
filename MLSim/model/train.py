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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 30000)')
    parser.add_argument('--resume', type=str, 
                        help="Model to resume training from",default=None)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Test a model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--lr', type=float, default=1e-4,
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
        embedconfig = EmbedConfig(74+39, 51, 256, 64, 128, 128, 512, 256, 512)
        transconfig = TransConfig(96, 512, 8, 2048, 4)
        model = TAO(embedconfig, transconfig)
    
    print(model)
    model.to(device)
    
    # return
    # 使用 DataLoader
    dataFloder = "/mnt/mix/ssd/shixl/TAO/MLSim/workload/spec2006/bzip2/trainData"
    # trainDataNames = ["data.txt0", "data.txt1", "data.txt2", "data.txt3", "data.txt4", "data.txt5", "data.txt6", "data.txt7", "data.txt8"]
    trainDataNames = ["data.txt9"]
    testDataNames = ["data.txt9"]
    
    trainDataFiles = []
    for dataName in trainDataNames:
        trainDataFiles.append(os.path.join(dataFloder, dataName))
    trainDataSet = CustomDataset(trainDataFiles)
    
    testDataFiles = []
    for dataName in testDataNames:
        testDataFiles.append(os.path.join(dataFloder, dataName))
    testDataSet = CustomDataset(testDataFiles)
    
            
    train_data_loader = DataLoader(trainDataSet, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_data_loader = DataLoader(testDataSet, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    mean = torch.tensor(np.array([7.33779, 413.29988])).to(device)
    std = torch.tensor(np.array([52.09697, 276.67064])).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 70, 90], 0.1)
    
    best, best_idx = [float('inf'), -1]
    
    for epoch in range(1, args.epochs + 1):
        train(args=args, 
              model=model,
              device=device, 
              train_loader=train_data_loader,
              optimizer=optimizer,
              epoch=epoch)
        scheduler.step()
        
        multiplier=1
        if (epoch % multiplier == 0):
              t = test(args=args, 
                       model=model, 
                       device=device,
                       test_loader=test_data_loader,
                       mean=mean,
                       std=std)
              if t < best:
                  print("Got improvement at {} (from {:.4f} to {:.4f}".format(epoch,best,t))
                  best , best_idx = (t,epoch)
                  if use_cuda:
                      torch.save(model, "models/best.pt" )
                  else:
                      torch.save(model, "best.pt" )
                    
        # if epoch > best_idx + 3*multiplier : #e.g. best is at 100, we exit at 401
        #     print("Ending at {} as there's been no improvement in 300 epochs. Best was at {}".format(epoch,best_idx))
        #     break

        if (args.save_model) and (epoch % 10 == 0):
            print("Saving")
            if use_cuda:
                torch.save(model,
                           "models/%d.pt" % epoch)
            else:
                torch.save(model,
                           "models/%d.pt" % epoch)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    print("Device:" ,device)
    loss = nn.MSELoss() 
#    loss = nn.L1Loss() #(reduction=None) 

    start_time = time.time()
    cur_time = start_time
    #print("Train loader has %d minibatches." % len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        loss_value = []
        def closure():
            optimizer.zero_grad()
            output = model(data)
            value = loss(output, target)
            value.backward()
            loss_value.append(value.item())
            return value

        #commentd out for lbfgs
        """optimizer.zero_grad()
        print("Forwarding model")
        output = model(data)
        value = loss(output,target) #/(target + 0.01)
        value.backward()"""

        
        optimizer.step(closure)
        prev_time = cur_time
        cur_time = time.time()
        if batch_idx%100 == 0:
            print("Epoch %d Minibatch %d took %f " \
                "Epoch of %d mbs would be %f "\
                "(Avg so far is %f)"\
                "loss is %f" % (epoch,
                                        batch_idx,
                                        cur_time - prev_time,
                                        len(train_loader),
                                        (cur_time - prev_time)*len(train_loader), 
                                        (cur_time-start_time)/(batch_idx + 1),
                                        loss_value[-1]))
            loss_value = []    
        
    print("Epoch %d  time was %f" % (epoch, time.time() - start_time))


def test(args, model, device, test_loader,mean,std):
    model.eval()
    loss_fetch = 0
    loss_completion = 0
    loss_total = 0
    correct = 0
    loss = nn.L1Loss(reduction='sum') 
    total_len = 0
    diff_fetch = 0

    with torch.no_grad():
        for data, target in test_loader:
            total_len += len(data)
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss_total += loss( output,target)
            
            norm_output = (output*std + mean)
            norm_target = (target*std + mean)
    
            loss_fetch += loss( norm_output[:,0:1],
                                norm_target[:,0:1] ).item()
            
            loss_completion += loss( norm_output[:,1:2],
                                     norm_target[:,1:2] ).item()
            
            norm_diff = norm_target - norm_output
            diff_fetch += torch.sum(norm_diff[:,0])

    loss_fetch /= total_len
    loss_completion /= total_len
    loss_total /= total_len
    print('Test set average loss fetch (cycles) {:.4f}, average loss completion (cycles) {:.4f}, average loss total {:.4f}, overall fetch diff {:.4f} '.format(loss_fetch,loss_completion,loss_total,diff_fetch))
    return loss_total.item()

if __name__ == '__main__':
    main()
    