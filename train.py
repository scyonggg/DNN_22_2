import argparse
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#from PIL import Image
import transforms 
#from torchvision import transforms
# from tensorboardX import SummaryWriter
from conf import settings
from utils import *
# from lr_scheduler import WarmUpLR
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from criterion import LSR

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='net type')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('--loss', type=str, default='label_smooth', choices=['label_smooth'], help='loss function')
    parser.add_argument('--weight_decay', action='store_true', help='1-D. No bias decay (regularization)')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'], help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.04, help='learning rate')
    parser.add_argument('--init_lr', type=float, default=0.001, help='initial learning rate when using learning rate scheduler')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='learning rate decay rate when using multi-step LR scheduler')
    parser.add_argument('--lr_scheduler', type=str, default='cosinelr', choices=['cosinelr', 'steplr'], help='learning rate scheduler')
    parser.add_argument('--epochs', type=int, default=450, help='training epoches')
    parser.add_argument('--warm_t', type=int, default=5, help='warm up phase')
    parser.add_argument('--decay_t', type=int, default=10, help='Decay LR for every decay_t epochs in StepLR')
    parser.add_argument('--gpus', type=str, default=0, help='gpu device')
    parser.add_argument('--log_step', type=int, default=1, help='printing loss step')
    parser.add_argument('--val_step', type=int, default=1, help='validation step')
    parser.add_argument('--save_step', type=int, default=1, help='save checkpoint step')
    parser.add_argument('--wandb', action='store_true', help='tracking with wandb')
    parser.add_argument('--run_name', type=str, default='scy_exp3', help='wandb run name')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    #checkpoint directory
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    #tensorboard log directory
    log_path = os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    if args.wandb:
        import wandb
        wandb.init(project='scy_test', entity="dnn_22_2", name=args.run_name, settings=wandb.Settings(code_dir="."))
        wandb.run.log_code(".")

    #get dataloader
    train_transforms = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToCVImage(),
        transforms.RandomResizedCrop(settings.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        #transforms.RandomErasing(),
        #transforms.CutOut(56),
        transforms.ToTensor(),
        transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
    ])

    test_transforms = transforms.Compose([
        transforms.ToCVImage(),
        transforms.CenterCrop(settings.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
    ])

    train_dataloader = get_train_dataloader(
        settings.DATA_PATH,
        train_transforms,
        args.batch_size,
        args.num_workers
    )

    test_dataloader = get_test_dataloader(
        settings.DATA_PATH,
        test_transforms,
        args.batch_size,
        args.num_workers
    )

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    net = get_network(args)
    net = init_weights(net)

    
    # if isinstance(args.gpus, int):
    #     args.gpus = [args.gpus]
    
    # net = nn.DataParallel(net, device_ids=args.gpus)
    net = net.cuda()

    #cross_entropy = nn.CrossEntropyLoss() 
    if args.loss == 'label_smooth':
        lsr_loss = LSR()  # Label smoothing
        
    #apply no weight decay on bias
    if args.weight_decay:
        params = split_weights(net)
    else:
        params = net.parameters()

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    if args.lr_scheduler == 'cosinelr':
        warmup_scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, warmup_t=args.warm_t, warmup_lr_init=args.init_lr)
    elif args.lr_scheduler == 'steplr':
        warmup_scheduler = StepLRScheduler(optimizer, decay_t=args.decay_t, warmup_t=args.warm_t, warmup_lr_init=args.init_lr, decay_rate=args.decay_rate)

    #set up training phase learning rate scheduler
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES)
    #train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warm)

    num_iters = len(train_dataloader)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        #training procedure
        net.train()
        
        for batch_index, (images, labels) in enumerate(train_dataloader):
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            predicts = net(images)
            loss = lsr_loss(predicts, labels)
            loss.backward()
            optimizer.step()

            n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1

            if batch_index % args.log_step == 0:
                print(f'Epoch : [{epoch} / {args.epochs}], \tIter : [{batch_index} / {num_iters}], \tLoss : {loss.item()}')
                if args.wandb:
                    wandb.log({'Epoch' : epoch, 'Iter' : batch_index, 'Train Loss': loss.item()})
            
        warmup_scheduler.step(epoch)
        if args.wandb:
            wandb.log({'LR' : optimizer.param_groups[0]['lr']})

        if epoch % args.val_step == 0:
            with torch.no_grad():
                net.eval()

                total_loss = 0
                correct = 0
                for images, labels in test_dataloader:

                    images = images.cuda()
                    labels = labels.cuda()

                    predicts = net(images)
                    _, preds = predicts.max(1)
                    correct += preds.eq(labels).sum().float()

                    loss = lsr_loss(predicts, labels)
                    total_loss += loss.item()

                test_loss = total_loss / len(test_dataloader)
                acc = correct / len(test_dataloader.dataset)
                print('Test set: loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, acc))
                if args.wandb:
                    wandb.log({'Validation Loss': test_loss, 'Validation acc': acc})

        #save weights file
        if epoch % args.save_step == 0:
            if best_acc < acc:
                print(f'Saving checkpoint ... accuracy = {acc}')
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                best_acc = acc
                continue
            
    










    


    

