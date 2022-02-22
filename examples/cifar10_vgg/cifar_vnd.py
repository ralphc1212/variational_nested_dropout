import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models.vgg
torch.backends.cudnn.deterministic = True

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os, sys
sys.path.append('../')

from models import VGG_VND
from modules.ngd import NGD
from modules.torch_vnd  import get_ard_reg_vnd
from modules.torch_vdo import LinearVDO, get_ard_reg_vdo

def set_track_running_stats(module, set):
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        module.track_running_stats = set
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_track_running_stats(submodule, set)

def set_weight_prob_fwd(module, set):
    if isinstance(module, abnn_eval.Conv2dVODO_eval) \
            or isinstance(module, abnn_eval.LinearVODO_eval) \
            or isinstance(module, nn_vdo.LinearVDO) \
            or isinstance(module, nn_vdo.Conv2dVDO):
        module.set_weight_prob_fwd(set)
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_weight_prob_fwd(submodule, set)

def set_ord_mask_ones_prop(module, set):
    if isinstance(module, abnn_eval.Conv2dVODO_eval) or isinstance(module, abnn_eval.LinearVODO_eval):
        module.set_ord_mask_ones_prop(set)
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_ord_mask_ones_prop(submodule, set)

def set_bn_train(module, train):
    if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        if train:
            module.train()
        else:
            module.eval()
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_bn_train(submodule, train)

MODEL = '11-wide'
SUFFIX = '-ngd'
reg_factor = 1e-5

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

folder = 'checkpoint/cifar_vnd.vgg'+MODEL+'.reg-5'+SUFFIX+'/'
ckpt_file = 'ckpt_abnn.ngd.pm'

best_acc = 0  # best test accuracy.
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=6)

testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=6)

# Model
print('==> Building model..')
model = VGG_VND(10, batch_norm=True, config=MODEL,
                   NUM_DIV=32, FREEZE_PART=2, PI=0.95).to(device)

param_first = []
param_last = []

for k,v in model.named_parameters():
    if 'features' in k:
        param_first.append(v)
    else:
        param_last.append(v)

BASE_LR = 1e-1
criterion = nn.CrossEntropyLoss()

# optim_f = optim.SGD(param_first, lr=BASE_LR, momentum=0.9)
# optim_l = optim.SGD(param_last, lr=BASE_LR, momentum=0.9)
optim_f = NGD(param_first, lr=BASE_LR, momentum=0.9)
optim_l = NGD(param_last, lr=BASE_LR, momentum=0.9)
sched_f = torch.optim.lr_scheduler.MultiStepLR(optim_f, milestones=[150,300,450,600], gamma=0.3)
sched_l = torch.optim.lr_scheduler.MultiStepLR(optim_l, milestones=[150,300,450,600], gamma=0.3)

tr_accs = []
te_accs = []

lrs = []

# Training
def train(epoch):

    global reg_factor

    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = []
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optim_f.zero_grad()
        optim_l.zero_grad()
        # nan fixed

        outputs = model(inputs)
        lld = criterion(outputs, targets)
        reg_first = get_ard_reg_vnd(model)
        reg_last = get_ard_reg_vdo(model)
        loss = lld + reg_factor * (reg_first + reg_last)
        # loss = lld + reg_factor * reg_first
        loss.backward()

        optim_f.step()
        optim_l.step()

        # if torch.sum(torch.isnan(lld)) != 0 or torch.sum(torch.isnan(reg_first)) != 0:
        if torch.sum(torch.isnan(lld)) != 0 or torch.sum(torch.isnan(reg_first)) != 0 or torch.sum(torch.isnan(reg_last)) != 0:

            print('outputs: ', outputs)
            print('lld: ', lld)
            print('reg_first: ', reg_first)
            print('reg_last: ', reg_last)

            state = {
                'net': model.state_dict(),
                'acc': correct * 100.0 / total,
                'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, 'checkpoint/ckpt_abnn.nan.pm')
            exit()

        train_loss.append(loss.detach().item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Train loss: %.2f' % np.mean(train_loss))
    print('Train accuracy: %.2f%%' % (correct * 100.0/total))

    return correct * 100.0 / total

def test(epoch):
    # model.eval()
    test_loss = []
    correct = 0
    total = 0

    global best_acc

    model.train()
    set_track_running_stats(model, False)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    set_track_running_stats(model, True)

    # Save checkpoint.
    acc = 100.*correct/total
    print('Test loss: %.2f' % np.mean(test_loss))
    print('Test accuracy: %.2f%%' % acc)
    print('Regularization factor: {}'.format(reg_factor))

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': correct * 100.0/total,
            'epoch': epoch,
        }
        if not os.path.isdir(folder):
            os.mkdir(folder)
        torch.save(state, folder+ckpt_file)
        best_acc = correct * 100.0/total

    if epoch % 100 == 0:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': correct * 100.0/total,
            'epoch': epoch,
        }
        if not os.path.isdir(folder):
            os.mkdir(folder)
        torch.save(state, folder + ckpt_file + '.epoch' + str(epoch))

    return 100.*correct/total


for epoch in range(start_epoch, start_epoch+600):
    tr_acc = train(epoch)
    te_acc = test(epoch)

    tr_accs.append(tr_acc)
    te_accs.append(te_acc)

    lr = 0.
    for group in optim_f.param_groups:
        lr = group['lr']
    lrs.append(lr)
    torch.save([tr_accs, te_accs, lrs], folder + 'cifar_vnd.wide.vgg' + MODEL + '.curve')

    sched_f.step()
    sched_l.step()
