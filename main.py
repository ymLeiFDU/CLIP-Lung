
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import yaml
import random
import configs
import datetime
import os
import argparse
from tqdm import tqdm
from terminaltables import AsciiTable

from models import *
from models.modules.ema import EMA
from utils.utils import save_args, Tee
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from data.dataset_lung import Lung


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if len(args.gpu) > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, epoch, model, ema, trainloader):
    model.train()
    train_loss = 0
    ce_loss, contrast_loss = 0, 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets, weights) in tqdm(enumerate(trainloader)):
        inputs, targets, weights = inputs.cuda(), targets.cuda(), weights.cuda()

        optimizer.zero_grad()
        if args.language:

                outputs, contrast_loss = model(
                                            inputs, args, targets, weights, mode = 'train')


        # Compute losses
        ce = criterion(outputs, targets)

        loss = args.lambda_ce * ce + args.lambda_contrast * contrast_loss
        loss.backward()
        optimizer.step()

        if args.use_ema:
            ema.update_params()

        train_loss += loss.item()
        ce_loss += ce.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct += predicted.eq(targets).sum().item()

    print('Train < Loss: {:3f} | Acc: {:3f} ({:d}/{:d}) >'.format(train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    if args.use_ema:
        ema.update_buffer()

    writer.add_scalar('Train/loss', train_loss, epoch)
    writer.add_scalar('Train/ce', ce_loss, epoch)
    writer.add_scalar('Train/Acc', 100.*correct/total, epoch)

    return model


def test(args, test_loader, epoch, model, ema):
    global best_acc
    global best_epoch
    global best_model
    model.eval()
    test_loss = 0
    ce_loss, contrast_loss = 0, 0
    correct = 0
    total = 0


    if args.use_ema:
        ema.apply_shadow()
        ema.model.eval()
        ema.model.cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets, weights) in tqdm(enumerate(test_loader)):
            inputs, targets, weights = inputs.cuda(), targets.cuda(), weights.cuda()

            if args.use_ema:
                if args.language:
                    outputs, contrast_loss = ema.model(
                            inputs, args, targets, weights, mode = 'test')
                else:
                    outputs, emb = ema.model(inputs, args)

            ce = criterion(outputs, targets)

            loss = ce

            test_loss += loss.item()
            ce_loss += ce.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Test < Loss: {:3f} | Acc: {:3f} ({:d}/{:d}) >'.format(test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    if args.use_ema:
        ema.restore()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_dir + '/' + args.ablation + '.pth')
        
        best_acc = acc
        best_model = model
        best_epoch = epoch

    writer.add_scalar('Test/loss', test_loss, epoch)
    writer.add_scalar('Test/ce', ce_loss, epoch)
    writer.add_scalar('Test/Acc', acc, epoch)


    if epoch == args.epochs - 1:
        torch.save(best_model, checkpoint_dir + '/trained_models' + '/Acc_{:.4f}_epoch_{}_model.pth'.format(best_acc, best_epoch))
        torch.save(best_model.state_dict(), checkpoint_dir + '/trained_models' + '/Acc_{:.4f}_epoch_{}_model_dict.pth'.format(best_acc, best_epoch))
    
    return best_acc, best_epoch
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lung-VLM')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--wd', default = 1e-4, type = float, help = 'weight_decay')
    parser.add_argument('--train-batch-size', '-train-bs', default = 128, type=int)
    parser.add_argument('--test-batch-size', '-test-bs', default = 64, type=int)
    parser.add_argument('--gpu', type = str, default = '0')
    parser.add_argument('--save-files', '-save', action = 'store_true')
    parser.add_argument('--model', type = str, default = 'resnet')
    parser.add_argument('--epochs', type = int, default = 200)
    parser.add_argument('--ablation', type = str, default = '2-class')
    parser.add_argument('--use-ema', '-ema', action = 'store_true')
    parser.add_argument('--ema-alpha', type = float, default = 0.999)
    parser.add_argument('--language', '-lang', action = 'store_true')
    parser.add_argument('--dataset', type = str, default = 'lico')

    parser.add_argument('--lambda-ce', type = float, default = 1.)
    parser.add_argument('--lambda-contrast', type = float, default = 1.)
    parser.add_argument('--seed', default=None, type=int, help="random seed")
    parser.add_argument('--workers', default=4, type=int)
    args = parser.parse_args()

    if args.seed > 0:
        set_seed(args)

    t = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    '''
    ==================== Saving files ====================
    '''
    with open('configs/paths.yaml', 'r') as f:
        paths = yaml.load(f, Loader=yaml.FullLoader)

    if args.save_files:
        project_dir = '/home/ymlei/Lung-VLM'
        if not os.path.exists(project_dir + '/ckpt/' + args.model):
            os.mkdir(project_dir + '/ckpt/' + args.model)

        checkpoint_dir = project_dir + '/ckpt/' + args.model + '/' + t + '_' + args.ablation

        print('Files saving dir: ', checkpoint_dir)
        files = paths['project_files']

        save_args(checkpoint_dir, files)
        logger = Tee(checkpoint_dir + '/log.txt', 'a')

        '''
        Tensorboard Summary
        '''
        summary = TensorboardSummary(checkpoint_dir)
        writer = summary.create_summary()



    best_acc = 0  
    start_epoch = 0  

    '''
    ==================== Datasets ====================
    '''


    print('==> Preparing data..')
    train_data =  # create your own dataset
    train_loader = DataLoader(
            train_data,
            batch_size = args.train_batch_size,
            shuffle = False,
            num_workers = args.workers
            )

    test_data = # create your own dataset
    test_loader = DataLoader(
            test_data,
            batch_size = args.test_batch_size,
            shuffle = False,
            num_workers = args.workers
            )
    


    '''
    ==================== Build Models ====================
    '''
    print('==> Building model..')
    num_classes = 2
    model = ResNet18(num_classes = num_classes, args = args)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu[0]))
        if len(args.gpu) > 1:
            model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpu.split(',')])
    model = model.cuda()


    '''
    ==================== Training Settings ====================
    '''
    if args.ema_alpha != 0:
        print('==> Training With EMA ...')
        ema = EMA(model, alpha = args.ema_alpha)
    else:
        print('==> Training NO EMA ...')



    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # scheduler = torch.optim.lr_scheduler.StepLR(argsimizer, step_size = 100, gamma = 0.1)

    '''
    ==================== Training ====================
    '''
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print('\n==> Epoch: %d / %d, Ablation: %s, %s' % (epoch, args.epochs, args.ablation, t))

        if args.use_ema:
            ema = ema
        else:
            ema = None

        model = train(args, epoch, model,
            ema = ema, 
            trainloader = train_loader)
        best_acc, best_epoch = test(args, test_loader, epoch, model, ema = ema, 
            )
        scheduler.step()

    print('Training done, best Acc: {:2f}, @ epoch {:d}'.format(best_acc, best_epoch))











