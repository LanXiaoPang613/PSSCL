from __future__ import print_function
import os
import matplotlib as mpl

# if os.environ.get('DISPLAY', '') == '':
#     print('no display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import argparse
import numpy as np
import torchvision.models as models
from VGG_animal import vgg19_bn
from PreResNet_clothing1M import resnet50
from sklearn.mixture import GaussianMixture
# import dataloader_food101N as dataloader
import dataloader_food101 as dataloader
import pdb
import io
import PIL
from torchvision import transforms
import seaborn as sns
import sklearn.metrics as metrics
import pickle
import json
import pandas as pd
import time
from pathlib import Path
from utils_plot import plot_guess_view, plot_histogram_loss_pred, \
    plot_model_view_histogram_loss, plot_model_view_histogram_pred, plot_tpr_fpr
import robust_loss, Contrastive_loss

sns.set()

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')

parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=160, type=int)
parser.add_argument('--id', default='food101N')
# parser.add_argument('--data_path', default='C:/Users/zhangqian/Desktop/all_datasets/Food-101N_release', type=str, help='path to dataset')
parser.add_argument('--data_path', default='C:/Users/zhangqian/Desktop/all_datasets/food-101', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=101, type=int)

parser.add_argument('--resume', default=False , type=bool, help='Resume from chekpoint')
parser.add_argument('--num_clean', default=5, type=int)
parser.add_argument('--run', default=0, type=int)
parser.add_argument('--yespenalty', default=1, type=int)
parser.add_argument('--dataset', default='food101N', type=str)


args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, savelog=False):
    net.train()
    net2.eval()  # fix one network and train the other

    train_loss = train_loss_lx = train_loss_u = train_loss_penalty = 0

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    max_iters = ((len(labeled_trainloader.dataset) + len(unlabeled_trainloader.dataset)) // args.batch_size) + 1

    cont_iters = 0

    while (cont_iters < max_iters):  # longmix
        for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader):
            try:
                inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.__next__()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.__next__()
            batch_size = inputs_x.size(0)

            if inputs_u.size(0) <=1 or batch_size <= 1:
                # Expected more than 1 value per channel when training, got input size torch.Size([1, 128])
                continue

            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
            w_x = w_x.view(-1, 1).type(torch.FloatTensor)

            inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()

            with torch.no_grad():
                # label co-guessing of unlabeled samples
                outputs_u11 = net(inputs_u)[1]
                outputs_u12 = net(inputs_u2)[1]
                outputs_u21 = net2(inputs_u)[1]
                outputs_u22 = net2(inputs_u2)[1]

                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)
                      + torch.softmax(outputs_u21,dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
                ptu = pu ** (1 / args.T)  # temparature sharpening

                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()

                # label refinement of labeled samples
                outputs_x = net(inputs_x)[1]
                outputs_x2 = net(inputs_x2)[1]

                px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                px = w_x * labels_x + (1 - w_x) * px
                ptx = px ** (1 / args.T)  # temparature sharpening

                targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                targets_x = targets_x.detach()

            ## Unsupervised Contrastive Loss， 两个strong_da用于对比学习，两个weak_da用于半监督
            f1, _ = net(inputs_u3)
            f2, _ = net(inputs_u4)
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            loss_simCLR = contrastive_criterion(features)

            # mixmatch
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)

            all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            logits = net(mixed_input)[1]
            logits_x = logits[:batch_size * 2]
            logits_u = logits[batch_size * 2:]

            Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u,
                                     mixed_target[batch_size * 2:], epoch + batch_idx / num_iter, warm_up)

            # regularization
            prior = torch.ones(args.num_class) / args.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = Lx + args.yespenalty * penalty+ args.lambda_c*loss_simCLR

            train_loss += loss
            train_loss_lx += Lx
            train_loss_u += Lu
            train_loss_penalty += penalty

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('\r')
            sys.stdout.write(
                '%s: | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f, cl loss: %.2f'
                % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter,
                   Lx.item(), Lu.item(), loss_simCLR.item()))
            sys.stdout.flush()
            cont_iters = cont_iters + 1
            if cont_iters == max_iters:
                break

    if savelog:
        train_loss /= len(labeled_trainloader.dataset)
        train_loss_lx /= len(labeled_trainloader.dataset)
        train_loss_u /= len(labeled_trainloader.dataset)
        train_loss_penalty /= len(labeled_trainloader.dataset)
        # Record training loss from each epoch into the writer
        # writer_tensorboard.add_scalar('Train/Loss', train_loss.item(), epoch)
        # writer_tensorboard.add_scalar('Train/Lx', train_loss_lx.item(), epoch)
        # writer_tensorboard.add_scalar('Train/Lu', train_loss_u.item(), epoch)
        # writer_tensorboard.add_scalar('Train/penalty', train_loss_penalty.item(), epoch)
        # writer_tensorboard.close()


use_robust = True


def warmup(epoch, net, optimizer, dataloader, savelog=False):
    net.train()
    wm_loss = 0
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)[1]
        if use_robust:
            loss = warm_criterion(outputs, labels)
            L = loss
        else:
            loss = CEloss(outputs, labels)
            L = loss

        wm_loss += L
        L.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s: | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()


best_acc = 86.06


def test(epoch, net1, net2):
    global best_acc
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)[1]
            outputs2 = net2(inputs)[1]
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            test_loss += CEloss(outputs1, targets)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    acc_hist.append(acc)
    if best_acc < acc:
        best_acc = acc
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%, Best_acc: %.2f%%\n" % (epoch, acc, best_acc))
    test_log.write('Epoch:%d   Accuracy:%.2f Best_acc: %.2f\n' % (epoch, acc, best_acc))
    test_log.flush()


def eval_train(model, all_loss, all_preds, all_hist, savelog=False):
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))
    preds = torch.zeros(len(eval_loader.dataset))
    preds_classes = torch.zeros(len(eval_loader.dataset), args.num_class)
    eval_loss = train_acc = 0
    # paths = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)[1]
            loss = CE(outputs, targets)
            eval_loss += CEloss(outputs, targets)

            _, pred = torch.max(outputs.data, -1)
            acc = float((pred == targets.data).sum())
            train_acc += acc
            eval_preds = F.softmax(outputs, -1).cpu().data

            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
                preds[index[b]] = eval_preds[b][targets[b]]
                preds_classes[index[b]] = eval_preds[b]
                # paths.append(path[b])

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)
    all_preds.append(preds)
    all_hist.append(preds_classes)

    input_loss = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]

    return prob, all_loss, all_preds, all_hist


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def get_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, args.num_class)
    return model


def create_model():
    model = resnet50(num_classes=args.num_class)
    # model = get_model()
    model = model.cuda()
    return model


def guess_unlabeled(net1, net2, unlabeled_trainloader):
    net1.eval()
    net2.eval()

    guessedPred_unlabeled = []
    for batch_idx, (inputs_u, inputs_u2, _, _) in enumerate(unlabeled_trainloader):
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net1(inputs_u)[1]
            outputs_u12 = net1(inputs_u2)[1]
            outputs_u21 = net2(inputs_u)[1]
            outputs_u22 = net2(inputs_u2)[1]

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21,
                                                                                                        dim=1) + torch.softmax(
                outputs_u22, dim=1)) / 4
            ptu = pu ** (1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            _, guessed_u = torch.max(targets_u, dim=-1)
            guessedPred_unlabeled.append(guessed_u)

    return torch.cat(guessedPred_unlabeled)


def save_models(epoch, net1, optimizer1, net2, optimizer2, save_path):
    state = ({
        'epoch': epoch,
        'state_dict1': net1.state_dict(),
        'optimizer1': optimizer1.state_dict(),
        'state_dict2': net2.state_dict(),
        'optimizer2': optimizer2.state_dict()

    })
    state2 = ({'all_loss': all_loss,
               'all_preds': all_preds,
               'hist_preds': hist_preds,
               'all_idx_view_labeled': all_idx_view_labeled,
               'all_idx_view_unlabeled': all_idx_view_unlabeled,
               'all_superclean': all_superclean,
               'acc_hist': acc_hist
               })
    state3 = ({
        'all_superclean': all_superclean
    })

    if epoch % 1 == 0:
        fn2 = os.path.join(save_path, 'model_ckpt.pth.tar')
        torch.save(state, fn2)
        fn3 = os.path.join(save_path, 'hcs_%s_cn%d_run%d.pth.tar' % (
        args.dataset, args.num_clean, args.run))
        torch.save(state3, fn3)

if __name__ == '__main__':

    name_exp = 'longremix_stage2_cn%d' % args.num_clean

    exp_str = '%s_%s_lu_%d' % (args.dataset, name_exp, int(args.lambda_u))
    if args.run > 0:
        exp_str = exp_str + '_run%d' % args.run
    path_exp = './checkpoint/' + exp_str

    path_plot = os.path.join(path_exp, 'plots')

    Path(path_exp).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(path_exp, 'savedDicts')).mkdir(parents=True, exist_ok=True)
    Path(path_plot).mkdir(parents=True, exist_ok=True)

    incomplete = os.path.exists("./checkpoint/%s/model_ckpt.pth.tar" % (exp_str))
    print('Incomplete...', incomplete)

    if incomplete == False:
        stats_log = open('./checkpoint/%s/%s' % (exp_str, args.dataset) + '_stats.txt',
                         'w')
        test_log = open('./checkpoint/%s/%s' % (exp_str, args.dataset) + '_acc.txt',
                        'w')
        time_log = open('./checkpoint/%s/%s' % (exp_str, args.dataset) + '_time.txt',
                        'w')
    else:
        stats_log = open('./checkpoint/%s/%s' % (exp_str, args.dataset) + '_stats.txt',
                         'a')
        test_log = open('./checkpoint/%s/%s' % (exp_str, args.dataset) + '_acc.txt',
                        'a')
        time_log = open('./checkpoint/%s/%s' % (exp_str, args.dataset) + '_time.txt',
                        'a')

    # writer_tensorboard = SummaryWriter('tensor_runs/'+exp_str)

    warm_up = 10

    loader = dataloader.food_dataloader(root=args.data_path, batch_size=args.batch_size,
                                         num_workers=0)

    print('| Building net')
    pretrained = True
    model = get_model()
    net1 = create_model()
    net2 = create_model()
    cudnn.benchmark = True

    criterion = SemiLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 30, 1e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, 30, 1e-4)

    # Copy Saved Data
    if pretrained == True:
        params = model.named_parameters()
        params1 = net1.named_parameters()
        params2 = net2.named_parameters()

        dict_params2 = dict(params2)
        dict_params1 = dict(params1)

        for name1, param in params:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(param.data)
                dict_params1[name1].data.copy_(param.data)

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    conf_penalty = NegEntropy()

    resume_epoch = 0

    if incomplete == True:
        print('loading Model...\n')
        load_path = 'checkpoint/%s/model_ckpt.pth.tar' % (exp_str)
        ckpt = torch.load(load_path)
        resume_epoch = ckpt['epoch']
        print('resume_epoch....', resume_epoch)
        net1.load_state_dict(ckpt['state_dict1'])
        net2.load_state_dict(ckpt['state_dict2'])
        optimizer1.load_state_dict(ckpt['optimizer1'])
        optimizer2.load_state_dict(ckpt['optimizer2'])

        all_superclean = [[], []]
        all_idx_view_labeled = [[], []]
        all_idx_view_unlabeled = [[], []]
        all_preds = [[], []]  # save the history of preds for two networks
        hist_preds = [[], []]
        acc_hist = []
        all_loss = [[], []]  # save the history of losses from two networks

        # all_idx_view_labeled = ckpt['all_idx_view_labeled']
        # all_idx_view_unlabeled = ckpt['all_idx_view_unlabeled']
        # all_preds = ckpt['all_preds']
        # hist_preds = ckpt['hist_preds']
        # acc_hist = ckpt['acc_hist']
        # all_loss = ckpt['all_loss']

        # superclean_path = os.path.join('hcs/', 'hcs_%s_%.2f_%s_cn%d_run%d.pth.tar' % (
        # args.dataset, args.r, args.noise_mode, args.num_clean, args.run))
        # ckpt = torch.load(superclean_path)
        # all_superclean = ckpt['all_superclean']
    else:
        all_superclean = [[], []]
        all_idx_view_labeled = [[], []]
        all_idx_view_unlabeled = [[], []]
        all_preds = [[], []]  # save the history of preds for two networks
        hist_preds = [[], []]
        acc_hist = []
        all_loss = [[], []]  # save the history of losses from two networks

    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')

    name_exp111 = 'longremix_stage1_cn%d' % args.num_clean
    exp_str111 = '%s_%s_lu_%d' % (args.dataset, name_exp111, int(args.lambda_u))
    ckpt_sc = torch.load(
        './checkpoint/%s/hcs_%s_cn%d_run%d.pth.tar' % (
            exp_str111, args.dataset, args.num_clean, args.run))
    all_superclean = ckpt_sc['all_superclean']

    total_time = 0
    warmup_time = 0

    maxsize = 0
    max_i = 0
    for i in range(1, 60):  # E/2,后半段
        size = len(all_superclean[0][-i])
        if size > maxsize:
            maxsize = size
            max_i = i

    print('max = %d, i=%d' % (maxsize, max_i))
    idx_superclean = all_superclean[0][-max_i]

    warm_criterion = robust_loss.GCELoss(args.num_class, gpu='0')
    contrastive_criterion = Contrastive_loss.SupConLoss()
    second_ind = True
    # if incomplete:
    #     test(resume_epoch, net1, net2)

    for epoch in range(resume_epoch+1, args.num_epochs + 1):
        lr = args.lr
        # if epoch > 0 and epoch%10 == 0:
        #     lr /= 10^int(epoch/30)  #20
        if 80 > epoch >= 40:
            lr /= 10
        elif 120 > epoch >= 80:
            lr /= 100
        elif epoch >= 120:
            lr /= 1000

        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr

        if epoch < warm_up:
            warmup_trainloader = loader.run('warmup')

            start_time = time.time()
            print('Warmup Net1')
            warmup(epoch, net1, optimizer1, warmup_trainloader, savelog=True)
            print('\nWarmup Net2')
            warmup(epoch, net2, optimizer2, warmup_trainloader, savelog=False)
            end_time = round(time.time() - start_time)
            total_time += end_time
            warmup_time += end_time

            # prob1, all_loss[0], all_preds[0], hist_preds[0] = eval_train(net1, all_loss[0], all_preds[0], hist_preds[0])
            # prob2, all_loss[1], all_preds[1], hist_preds[1] = eval_train(net1, all_loss[1], all_preds[1], hist_preds[1])
            #
            # pred1 = (prob1 > args.p_threshold)
            # pred2 = (prob2 > args.p_threshold)
            # idx_view_labeled = (pred1).nonzero()[0]
            # idx_view_unlabeled = (1 - pred1).nonzero()[0]
            # all_idx_view_labeled[0].append(idx_view_labeled)
            # all_idx_view_labeled[1].append((pred2).nonzero()[0])
            # all_idx_view_unlabeled[0].append(idx_view_unlabeled)
            # all_idx_view_unlabeled[1].append((1 - pred2).nonzero()[0])

            if epoch == (warm_up - 1):
                time_log.write('Warmup: %f \n' % (warmup_time))
                time_log.flush()
        else:
            start_time = time.time()
            prob1, all_loss[0], all_preds[0], hist_preds[0] = eval_train(net1, all_loss[0], all_preds[0],
                                                                         hist_preds[0], savelog=True)

            prob2, all_loss[1], all_preds[1], hist_preds[1] = eval_train(net2, all_loss[1], all_preds[1],
                                                                         hist_preds[1], savelog=False)

            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)

            # Update probabilities
            prob1[idx_superclean] = 1
            prob2[idx_superclean] = 1

            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)  # co-divide
            train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, savelog=True)  # train net1

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
            train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader,
                  savelog=False)  # train net2
            end_time = round(time.time() - start_time)
            total_time += end_time

        save_models(epoch, net1, optimizer1, net2, optimizer2, path_exp)

        test(epoch, net1, net2)
        # scheduler1.step()
        # scheduler2.step()

    test_log.write('\nBest:%.2f  avgLast10: %.2f\n' % (max(acc_hist), sum(acc_hist[-10:]) / 10.0))
    test_log.close()

    time_log.write('SSL Time: %f \n' % (total_time - warmup_time))
    time_log.write('Total Time: %f \n' % (total_time))
    time_log.close()

