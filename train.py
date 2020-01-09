from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import os
import argparse
from read_Data import ImageNetData
import se_resnet
from train_val_txt import train_valtxt


def getline(the_file_path, line_number):
  if line_number < 1:
    return ''
  for cur_line_number, line in enumerate(open(the_file_path, 'rU')):
    if cur_line_number == line_number-1:
      return line
  return ''

def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes):
    since = time.time()
    resumed = False
    train_loss=0.6
    best_model_wts = model.state_dict()

    for epoch in range(args.start_epoch+1,num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                if args.start_epoch > 0 and (not resumed):
                    scheduler.step(args.start_epoch+1)
                    resumed = True
                else:
                    scheduler.step(train_loss)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            tic_batch = time.time()
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloders[phase]):
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                correct = preds.eq(labels).sum().float().item()
                running_corrects += correct

                batch_loss = running_loss / ((i+1)*args.batch_size)
                batch_acc = running_corrects / ((i+1)*args.batch_size)

                if phase == 'train' and i%args.print_freq == 0:
                    print('[Epoch {}/{}]-[batch:{}/{}]  {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                          epoch, num_epochs - 1, i, round(dataset_sizes[phase]/args.batch_size)-1,  phase, batch_loss, batch_acc, \
                        args.print_freq/(time.time()-tic_batch)))
                    tic_batch = time.time()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'train':
                train_loss = running_loss / dataset_sizes['train']

        if (epoch + 1) % args.save_epoch_freq == 0:
            torch.save(model.state_dict(), 'epoch' + str(epoch) + '_' + 'params.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model, 'model.pth')
    return model

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
        parser.add_argument('--data-dir', type=str, default=os.getcwd())
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--num-class', type=int, default=200)
        parser.add_argument('--num-epochs', type=int, default=200)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--num-workers', type=int, default=0)
        parser.add_argument('--gpus', type=str, default=0)
        parser.add_argument('--print-freq', type=int, default=10)
        parser.add_argument('--save-epoch-freq', type=int, default=1)
        parser.add_argument('--save-path', type=str, default="output")
        parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
        parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
        parser.add_argument('--network', type=str, default="se_resnet_18", help="")
        args = parser.parse_args()

        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        device_ids = [0, 1, 2]

        train_valtxt()
        # read data
        dataloders, dataset_sizes = ImageNetData(args)

        # use gpu or not
        use_gpu = torch.cuda.is_available()
        print("use_gpu:{}".format(use_gpu))

        # get model

        script_name = '_'.join([args.network.strip().split('_')[0], args.network.strip().split('_')[1]])

        if script_name == "se_resnet":
            model = getattr(se_resnet ,args.network)(num_classes = args.num_class)

        else:
            raise Exception("Please give correct network name such as se_resnet_xx or se_rexnext_xx")

        if args.resume:
            if os.path.isfile(args.resume):
                print(("=> loading checkpoint '{}'".format(args.resume)))
                checkpoint = torch.load(args.resume)
                base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
                model.load_state_dict(base_dict)
            else:
                print(("=> no checkpoint found at '{}'".format(args.resume)))
        device = torch.device("cuda:0")
        if use_gpu:
            model = model.cuda(device_ids[0])
            model = torch.nn.DataParallel(model, device_ids=device_ids).to(device)

        # define loss function
        criterion = nn.CrossEntropyLoss()

        optimizer_ft = optim.Adam(model.parameters(), lr=args.lr)

        exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=8,
                                                                      verbose=True,
                                                                      min_lr=0.0001,
                                                                      eps=1e-08)

        model = train_model(args=args,
                            model=model,
                            criterion=criterion,
                            optimizer=optimizer_ft,
                            scheduler=exp_lr_scheduler,
                            num_epochs=args.num_epochs,
                            dataset_sizes=dataset_sizes,)

