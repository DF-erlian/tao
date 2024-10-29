import argparse
import torch
import torch.nn as nn
import torch.nn.function as F
import torch.optim as optim

from model import TAOModel
from dataloader import CustomDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                        help='number of epochs to train (default: 30000)')
    parser.add_argument('--resume', type=str, 
                        help="Model to resume training from",default=None)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Test a model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    print("Batch sz is %d. Save? %s. Cuda? %s. Dev count: %d \n" % 
          (args.batch_size,
           str(args.save_model),
           str(use_cuda),
           torch.cuda.device_count()))
    
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: " ,device)
    
    # 使用 DataLoader
    data_file = '/worksapce/TAO/MLSim/workload/hello/trainData.txt'  # 数据文件路径
    dataset = CustomDataset(data_file)
    
    if args.resume:
        print("resuming")
        model = torch.load(args.resume, map_location=device)
        model.eval()
    else:
        print("starting from scratch")
        config = Config(512, 64, 8, 8, 2048, 96, 96)
        model = TAOModel(config)
    
    model.to(device)
    
    # if args.test:
        
    train_data_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    test_data_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best, best_idx = [float('inf'), -1]
    
    
    for epoch in range(1, args.epochs + 1):
        train(args=args, 
              model=model,
              device=device, 
              train_loader=train_data_loader,
              optimizer=optimizer,
              epoch=epoch)
        multiplier=100
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
                    
        if epoch > best_idx + 3*multiplier : #e.g. best is at 100, we exit at 401
            print("Ending at {} as there's been no improvement in 300 epochs. Best was at {}".format(epoch,best_idx))
            break

        if (args.save_model) and (epoch % 100 == 0):
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

        def closure():
            optimizer.zero_grad()
            output = model(data)
            value = loss(output, target)
            value.backward()
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
        """ print("Epoch %d Minibatch %d took %f " \
              "Epoch of %d mbs would be %f "\
              "(Avg so far is %f)" % (epoch,
                                      batch_idx,
                                      cur_time - prev_time,
                                      len(train_loader),
                                      (cur_time - prev_time)*len(train_loader), 
                                      (cur_time-start_time)/(batch_idx + 1))) """
        
    print("Epoch %d  time was %f" % (epoch, time.time() - start_time))


def test(args, model, device, test_loader,mean,std):
    model.eval()
    loss_fetch = 0
    loss_completion = 0
    loss_total = 0
    correct = 0
    loss = nn.L1Loss(reduction='sum') 
    total_len = 0

    with torch.no_grad():
        for data, target in test_loader:
            total_len += len(data)
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss_total += loss( output,target)
            
            norm_output = (output*std + mean)
            norm_target = (target*std + mean)
    
            loss_fetch += loss( norm_output[:,0:1].round(),
                                norm_target[:,0:1].round() ).item()
            
            loss_completion += loss( norm_output[:,1:2].round(),
                                     norm_target[:,1:2].round() ).item()
            

    loss_fetch /= total_len
    loss_completion /= total_len
    loss_total /= total_len
    print('Test set average loss fetch (cycles) {:.4f},  completion (cycles) {:.4f} NN obj {:.4f} '.format(loss_fetch,loss_completion,loss_total))
    return loss_total.item()

if __name__ == '__main__':
    main()
    