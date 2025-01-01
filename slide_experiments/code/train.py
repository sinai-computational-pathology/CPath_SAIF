import datasets
import modules
import os
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pdb
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()

#I/O PARAMS
parser.add_argument('--output', type=str, default='.', help='name of output directory')
parser.add_argument('--log', type=str, default='convergence.csv', help='name of log file')
parser.add_argument('--class_only', type=int, default=None, help='only class to use')
parser.add_argument('--aggregator', type=str, default='GMA', help='which aggregator to use')
parser.add_argument('--encoder', type=str, default='', choices=[
    'SP21M',
    'SP85M',
    'uni',
    'virchow',
    'virchow2',
    'gigapath',
], help='which encoder to use')
#OPTIMIZATION PARAMS
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled with the batch size, and specified here for a reference batch size of 256.""")
parser.add_argument('--lr_end', type=float, default=1e-6, help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
parser.add_argument("--warmup_epochs", default=10, type=int, help="Number of epochs for the linear learning-rate warm up.")
parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
parser.add_argument('--nepochs', type=int, default=40, help='number of epochs (default: 40)')
parser.add_argument('--workers', default=10, type=int, help='number of data loading workers (default: 10)')
parser.add_argument('--accumulation_steps', default=1, type=int, help='step to accumualate loss')
parser.add_argument('--data_version', required=True, type=str, help="The dataset version to use")
parser.add_argument('--organ', type=str, required=True, help='The organ to process (e.g., "skin", "neuro", "breast", "prostate").')
parser.add_argument('--shuffle_target', action='store_true', help="Shuffle the target column if specified")

def main():
    
    # Get user input
    global args
    args = parser.parse_args()

    # Set datasets
    train_dset, val_dset, weights = datasets.get_datasets(encoder=args.encoder, task='classification', class_only=args.class_only, data_version=args.data_version, organ=args.organ, shuffle_target=args.shuffle_target, random_seed=1234)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=args.workers)
    
    # Dim of features
    if args.encoder == 'ctranspath':
        args.ndim = 768
    elif args.encoder == 'phikon':
        args.ndim = 768
    elif args.encoder == 'uni':
        args.ndim = 1024
    elif args.encoder == 'virchow':
        args.ndim = 2560
    elif args.encoder == 'virchow2':
        args.ndim = 2560
    elif args.encoder == 'gigapath':
        args.ndim = 1536
    elif args.encoder == 'h-optimus-0':
        args.ndim = 1536
    elif args.encoder == 'SP21M':
        args.ndim = 384
    elif args.encoder == 'SP85M':
        args.ndim = 768
    
    # Get model
    if args.aggregator == 'GMA':
        gma = modules.GMA(dropout=True, n_classes=5, n_tasks=1, ndim=args.ndim)
    elif args.aggregator == 'GMA_multiple':
        gma = modules.GMA_multiple(dropout=True, n_classes=5, n_tasks=5, ndim=args.ndim)
    else:
        raise ("aggregator is not available")
    gma.cuda()
    
    # Set loss
    criterion = nn.CrossEntropyLoss(weight=weights).cuda()
    
    # Set optimizer
    params_groups = get_params_groups(gma)
    optimizer = optim.AdamW(params_groups)
    
    # Set schedulers
    lr_schedule = cosine_scheduler(
        args.lr,
        args.lr_end,
        args.nepochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.nepochs,
        len(train_loader),
    )
    cudnn.benchmark = True
        
    # Set output files
    with open(os.path.join(args.output,args.log), 'w') as fconv:
        fconv.write('epoch,metric,value\n')
    
    # Main training loop
    for epoch in range(args.nepochs+1):
        ## Training logic
        if epoch > 0:
            loss = train(epoch, train_loader, gma, criterion, optimizer, lr_schedule, wd_schedule, accumulation_steps=args.accumulation_steps)
            print('Training\tEpoch: [{}/{}]\tLoss: {}'.format(epoch, args.nepochs, loss))
            with open(os.path.join(args.output,args.log), 'a') as fconv:
                fconv.write('{},loss,{}\n'.format(epoch, loss))
        
        ## Validation logic
        probs = test(epoch, val_loader, gma)
        auc_ovr = roc_auc_score(val_loader.dataset.df.target, probs, multi_class='ovr')
        auc_ovo = roc_auc_score(val_loader.dataset.df.target, probs, multi_class='ovo')
        ### Printing stats
        print('Validation\tEpoch: [{}/{}]\tAUC_ovr: {}'.format(epoch, args.nepochs, auc_ovr))
        print('Validation\tEpoch: [{}/{}]\tAUC_ovo: {}'.format(epoch, args.nepochs, auc_ovo))
        with open(os.path.join(args.output,args.log), 'a') as fconv:
            fconv.write('{},auc_ovr,{}\n'.format(epoch, auc_ovr))
            fconv.write('{},auc_ovo,{}\n'.format(epoch, auc_ovo))
        
        if epoch == args.nepochs: # only save checkpoint for last epoch
            ### Model saving logic
            obj = {
                'epoch': epoch,
                'state_dict': gma.state_dict(),
                'auc_ovr': auc_ovr,
                'auc_ovo': auc_ovo,
                'optimizer' : optimizer.state_dict()
            }

            torch.save(obj, os.path.join(args.output,f'checkpoint_{epoch}.pth'))

def test(run, loader, model):
    # Set model in test mode
    model.eval()
    # Initialize probability vector
    probs = torch.zeros(len(loader.dataset), 5).cuda()
    # Loop through batches
    with torch.no_grad():
        for i, (input, _) in enumerate(loader):
            print('Inference\tEpoch: [{}/{}]\tBatch: [{}/{}]'.format(run, args.nepochs, i+1, len(loader)))
            ## Copy batch to GPU
            input = input.squeeze(0).cuda()
            ## Forward pass
            output = model(input)
            output = F.softmax(output, dim=1)
            ## Clone output to output vector
            probs[i] = output.detach().cpu()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer, lr_schedule, wd_schedule, accumulation_steps=1):
    # Set model in training mode
    model.train()
    # Initialize loss
    running_loss = 0.
    # Loop through batches
    for i, (input, target) in enumerate(loader):
        ## Update weight decay and learning rate according to their schedule
        it = len(loader) * (run-1) + i # global training iteration
        for j, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if j == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        
        ## Copy to GPU
        input = input.squeeze(0).cuda()
        target = target.long().cuda()
        ## Forward pass
        output = model(input)
        ## Calculate loss
        loss = criterion(output, target)
        loss.backward()
        running_loss += loss.item()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print('Training\tEpoch: [{}/{}]\tBatch: [{}/{}]\tLoss: {}'.format(
                run, args.nepochs, i + 1, len(loader), running_loss / (i + 1)))
        
    # After the last batch, make sure to step if there are remaining gradients
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return running_loss / len(loader)

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

if __name__ == '__main__':
    main()
