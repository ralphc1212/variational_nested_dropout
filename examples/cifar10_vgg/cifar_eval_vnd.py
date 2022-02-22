import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

import os, sys
sys.path.append('../')
import time
torch.backends.cudnn.deterministic = True

from models import VGG_VND_eval, VGG_VND
from modules import torch_vnd_eval, torch_vdo
from modules.calibration import ECELoss
import torchvision
from sklearn.metrics import roc_auc_score, average_precision_score


def set_track_running_stats(module, set):
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        module.track_running_stats = set
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_track_running_stats(submodule, set)

def set_weight_prob_fwd(module, set):
    if isinstance(module, torch_vnd_eval.Conv2dVND_eval) \
            or isinstance(module, torch_vdo.LinearVDO) \
            or isinstance(module, torch_vdo.Conv2dVDO):
        module.set_weight_prob_fwd(set)
    if hasattr(module, 'children'):
        for submodule in module.children():
            set_weight_prob_fwd(submodule, set)

def set_ord_mask_ones_prop(module, set):
    if isinstance(module, torch_vnd_eval.Conv2dVND_eval):
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ckpt_baseline_file = 'checkpoint/ckpt_baseline.abnn'

folder = 'checkpoint/cifar_vnd.vgg11-wide.reg-5-ngd/'
ckpt_file = 'ckpt_abnn.ngd.pm'


best_acc = 0  # best test accuracy
best_compression = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
reg_factor = 1e-5

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

n_classes = 10

# Model
print('==> Building model..')

model = VGG_VND_eval(n_classes, batch_norm = True, config=MODEL,
                        NUM_DIV=32, FREEZE_PART=2, PI=0.8, multiple=1.).to(device)
cab = ECELoss()

# if os.path.isfile(ckpt_file):
state_dict = model.state_dict()
checkpoint = torch.load(folder+ckpt_file,map_location='cpu')
state_dict.update(checkpoint['net'])
model.load_state_dict(state_dict,strict=False)
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

del checkpoint

criterion = nn.CrossEntropyLoss()

def test(train, weight_prob_fwd, prop, n_samples, collect=False, collect_epoch=2, ood_detection = False, ooddataset='SVHN'):
    if ood_detection:
        if ooddataset == 'SVHN':
            oodset = torchvision.datasets.SVHN(root='~/data', split='train', download=True, transform=transform_test)
            oodloader = torch.utils.data.DataLoader(oodset, batch_size=64, shuffle=False, num_workers=4)
        elif ooddataset == 'LSUN':
            oodset = torchvision.datasets.LSUN(root='~/data', classes='test',  transform=transform_test)
            oodloader = torch.utils.data.DataLoader(oodset, batch_size=64, shuffle=False, num_workers=4)

    def collect_stats(epoch):
        print('---collecting BN stats---')
        for epoch_i in range(epoch):
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    out = model(inputs)

    def retrieve_results():
        test_loss = []
        correct = 0
        total = 0
        calibs = []
        if ood_detection:
            cat_entros = []
            cat_maxps = []
            ens_disp = []

        # inference_time_seconds = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                # inputs, targets = inputs.view(inputs.shape[0], -1).to(device), targets.to(device)
                inputs, targets = inputs.to(device), targets.to(device)

                loss_per_x = []
                # start_ts = time.time()
                outputs = []
                for ith in range(n_samples):
                    out = model(inputs)
                    # inference_time_seconds += time.time() - start_ts
                    loss = criterion(out, targets)
                    loss_per_x.append(loss)
                    # outputs += out
                    outputs.append(out)

                outputs = torch.stack(outputs).permute(1,0,2)
                mean = torch.mean(outputs,dim=1)

                if torch.isnan(mean).sum() > 0:
                    print('is nan')
                    exit()
                calibration = cab(mean, targets)
                calibs.append(calibration)
                _, predicted = mean.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss = torch.mean(torch.tensor(loss_per_x))
                test_loss.append(loss.item())

                if ood_detection:
                    particles = torch.softmax(outputs, dim=2)
                    pt_mean = torch.mean(particles, dim=1)

                    cat_entros.append((-torch.bmm(pt_mean.view(pt_mean.shape[0], 1, -1),
                                            torch.log(pt_mean.view(pt_mean.shape[0], -1, 1)))).view(-1).cpu().numpy())
                    cat_maxps.append(torch.max(pt_mean, dim=1)[0].cpu().numpy())


            calibration = torch.mean(torch.tensor(calibs))

            if not ood_detection:
                return correct, total, calibration
            else:
                for batch_idx, (inputs, targets) in enumerate(oodloader):
                    inputs, targets = inputs.to(device), targets.to(device)

                    # start_ts = time.time()
                    outputs = []
                    for ith in range(n_samples):
                        out = model(inputs)
                        outputs.append(out)

                    outputs = torch.stack(outputs).permute(1,0,2)
                    mean = torch.mean(outputs,dim=1)

                    if torch.isnan(mean).sum() > 0:
                        print('is nan')
                        exit()

                    particles = torch.softmax(outputs, dim=2)
                    pt_mean = torch.mean(particles, dim=1)

                    cat_entros.append((-torch.bmm(pt_mean.view(pt_mean.shape[0], 1, -1),
                                            torch.log(pt_mean.view(pt_mean.shape[0], -1, 1)))).view(-1).cpu().numpy())
                    cat_maxps.append(torch.max(pt_mean, dim=1)[0].cpu().numpy())

                cat_entros = np.concatenate(cat_entros)
                cat_maxps = 1./np.concatenate(cat_maxps)

                indomain_or_not = np.concatenate([np.zeros(testloader.dataset.__len__()), np.ones(oodloader.dataset.__len__())])
                roc_entro = roc_auc_score(indomain_or_not, cat_entros)
                roc_maxp = roc_auc_score(indomain_or_not, cat_maxps)
                pr_entro = average_precision_score(indomain_or_not, cat_entros)
                pr_maxp = average_precision_score(indomain_or_not, cat_maxps)

                print("AUROC (cat_entros): ", roc_entro)
                print("AUROC (cat_maxps): ", roc_maxp)
                print("AUPR  (cat_entros): ", pr_entro)
                print("AUPR  (cat_maxps): ", pr_maxp)

                ood_results = {
                    'ROC-entro': roc_entro,
                    'ROC-maxp':  roc_maxp,
                    'PR-entro': pr_entro,
                    'PR-maxp': pr_maxp,
                    'var': cat_entros,
                }
                return ood_results, correct, total, calibration

    if train:
        model.train()
    else:
        model.eval()

        if isinstance(prop, list):
            model.set_specified_masks(prop)
        else:
            set_ord_mask_ones_prop(model, prop)

        set_weight_prob_fwd(model, weight_prob_fwd)

        if not weight_prob_fwd:
            n_samples = 1

    if collect:
        set_bn_train(model, True)
        collect_stats(epoch=collect_epoch)
        set_bn_train(model, False)

    if not ood_detection:
        correct, total, calibration = retrieve_results()
        acc = 100.*correct/total
        return None, acc, calibration.item()
    else:
        ood_results, correct, total, calibration = retrieve_results()
        acc = 100.*correct/total
        return ood_results, acc, calibration.item()


# test different masks & weight probabilistic forward
n_ranges = 15
ood_detection = True
results = {}
for i in range(n_ranges+1):
    prop = 1. - (1. / n_ranges) * i
    ood, acc, calibration = test(train=False, weight_prob_fwd=True, prop=prop, n_samples=8, collect=True, collect_epoch=1, ood_detection=ood_detection)

    width = float('%0.2f'%(1. - (1. / n_ranges) * i))
    acc = float('%0.3f'%(acc))
    calibration = float('%0.3f'%(calibration))
    print('Width: {}. Acc: {}. Calibration: {}.'.format(width, acc, calibration))
    results[width] = (acc, calibration, ood)
