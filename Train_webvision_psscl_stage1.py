from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import sys
import argparse
import numpy as np
from InceptionResNetV2 import *
from sklearn.mixture import GaussianMixture
import dataloader_webvision as dataloader
import torchnet
# from Contrastive_loss import *
import robust_loss, Contrastive_loss
from pathlib import Path
import time

parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')   ## Set the learning rate to 0.005 for faster training at the beginning
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')

parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=120, type=int)
parser.add_argument('--id', default='',type=str)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--data_path', default='C:/Users/Administrator/Desktop/DatasetAll/WebVision1.0/', type=str, help='path to dataset')
parser.add_argument('--resume', default=False , type=bool, help='Resume from chekpoint')
parser.add_argument('--dataset', default='WebVision', type=str)

parser.add_argument('--num_clean', default=5, type=int)
parser.add_argument('--run', default=0, type=int)
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')

args = parser.parse_args()
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# contrastive_criterion = SupConLoss()


## Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net2.eval()         # Fix one network and train the other    
    net.train()       

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter_lab = (len(labeled_trainloader.dataset)//args.batch_size)+1

    loss_x = 0
    loss_u = 0
    loss_scl = 0
    loss_ucl = 0

    num_iter = num_iter_lab

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
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)
        w_x = w_x.view(-1,1).type(torch.FloatTensor)

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()

        with torch.no_grad():

            # Label co-guessing of unlabeled samples
            _, outputs_u11 = net(inputs_u3)
            _, outputs_u12 = net(inputs_u4)
            _, outputs_u21 = net2(inputs_u3)
            _, outputs_u22 = net2(inputs_u4)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)
                  + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu**(1/args.T)                             ## Temparature Sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)   ## Normalize
            targets_u = targets_u.detach()

            ## Label refinement of labeled samples
            _, outputs_x  = net(inputs_x3)
            _, outputs_x2 = net(inputs_x4)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

            px = w_x*labels_x + (1-w_x)*px
            ptx = px**(1/args.T)                            ## Temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  ## normalize
            targets_x = targets_x.detach()

        ## Mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l,1-l)

        ## Unsupervised Contrastive Loss
        f1, _ = net(inputs_u)
        f2, _ = net(inputs_u2)
        f1    = F.normalize(f1, dim=1)
        f2    = F.normalize(f2, dim=1)
        features    = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss_simCLR = contrastive_criterion(features)


        all_inputs  = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a , input_b   = all_inputs , all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        ## Mixing inputs
        mixed_input  = (l * input_a[: batch_size * 2] + (1 - l) * input_b[: batch_size * 2])
        mixed_target = (l * target_a[: batch_size * 2] + (1 - l) * target_b[: batch_size * 2])

        _, logits = net(mixed_input)

        Lx = -torch.mean(
            torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1)
        )

        ## Regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx  + args.lambda_c*loss_simCLR + penalty
        loss_x += Lx.item()
        loss_ucl += loss_simCLR.item()

        ## Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s: | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Contrative Loss:%.4f'
                %(args.dataset,  epoch, args.num_epochs, batch_idx+1, num_iter, loss_x/(batch_idx+1), loss_ucl/(batch_idx+1)))
        sys.stdout.flush()


use_robust = True
def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)[1]               
        # loss = CEloss(outputs, labels)
        if use_robust:
            # loss = warm_criterion(outputs, labels)
            loss = CEloss(outputs, labels)
            L = loss
        else:
            loss = CEloss(outputs, labels)
            penalty = 0.#conf_penalty(outputs)
            L = loss + penalty

        L.backward()  
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:| Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()
        
        
def test(epoch,net1,net2,test_loader):
    acc_meter.reset()
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)[1]
            outputs2 = net2(inputs)[1]           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)                 
            acc_meter.add(outputs,targets)
    accs = acc_meter.value()
    return accs

def eval_train(model, all_loss, all_preds, all_hist, savelog=False):
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))
    # losses = torch.full(len(eval_loader.dataset), 100.)
    preds = torch.zeros(len(eval_loader.dataset))
    preds_classes = torch.zeros(len(eval_loader.dataset), args.num_class)
    eval_loss = train_acc = 0
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
    # max_losses = losses.max()
    # for i in range(len(losses)):
    #     if losses[i] == 0.:
    #         losses[i] = max_losses
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
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = InceptionResNetV2(num_classes=args.num_class)
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
            outputs_u11 = net1(inputs_u)
            outputs_u12 = net1(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)
                  + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
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
    # state2 = ({'all_loss': all_loss,
    #            'all_preds': all_preds,
    #            'hist_preds': hist_preds,
    #            'inds_clean': inds_clean,
    #            'inds_noisy': inds_noisy,
    #            'clean_labels': clean_labels,
    #            'noisy_labels': noisy_labels,
    #            'all_idx_view_labeled': all_idx_view_labeled,
    #            'all_idx_view_unlabeled': all_idx_view_unlabeled,
    #            'all_superclean': all_superclean,
    #            'acc_hist': acc_hist
    #            })
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
    name_exp = 'psscl_stage1_cn%d' % args.num_clean

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


    warm_up = 15
    mid_warmup = 25
    loader = dataloader.webvision_dataloader(batch_size=args.batch_size,num_workers=5,root_dir=args.data_path,
                                             log=stats_log, num_class=args.num_class)

    print('| Building net')
    net1 = create_model()
    net2 = create_model()
    cudnn.benchmark = True

    criterion = SemiLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)#1e-4
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    conf_penalty = NegEntropy()

    resume_epoch = 0
    net1 = nn.DataParallel(net1)
    net2 = nn.DataParallel(net2)
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
        all_loss = [[], []]
        start_epoch = resume_epoch+1
    else:
        all_superclean = [[], []]
        all_idx_view_labeled = [[], []]
        all_idx_view_unlabeled = [[], []]
        all_preds = [[], []]  # save the history of preds for two networks
        hist_preds = [[], []]
        acc_hist = []
        all_loss = [[], []]  # save the history of losses from two networks
        start_epoch = 0

    folder = 'Webvision_psscl'
    model_save_loc = './checkpoint/' + folder
    if not os.path.exists(model_save_loc):
        os.mkdir(model_save_loc)

    total_time = 0
    warmup_time = 0
    acc_hist = []

    warm_criterion = robust_loss.GCELoss(args.num_class, gpu='0')
    contrastive_criterion = Contrastive_loss.SupConLoss()
    second_ind = True

    # net1 = nn.DataParallel(net1)
    # net2 = nn.DataParallel(net2)

    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
    SR = 0
    best_acc = 0

    temp_n_clean = 2

    eval_loader = loader.run(0.5, 'eval_train')
    web_valloader = loader.run(0.5, 'test')
    imagenet_valloader = loader.run(0.5, 'imagenet')
    num_samples = len(eval_loader.dataset)
    print("Total Number of Samples: ", num_samples)
    if start_epoch > 0:
        web_acc = test(start_epoch, net1, net2, web_valloader)
        imagenet_acc = test(start_epoch, net1, net2, imagenet_valloader)
        print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n" % (
        start_epoch, web_acc[0], web_acc[1], imagenet_acc[0], imagenet_acc[1]))

    for epoch in range(start_epoch, args.num_epochs+1):
        # Manually Changing the learning rate ###
        lr=args.lr
        # if epoch >= 50:
        #     lr /= 10
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr

        if epoch<warm_up:
            warmup_trainloader = loader.run(0.5, 'warmup')

            start_time = time.time()
            print('Warmup Net1')
            warmup(epoch,net1,optimizer1,warmup_trainloader)
            print('\nWarmup Net2')
            warmup(epoch,net2,optimizer2,warmup_trainloader)

            end_time = round(time.time() - start_time)
            total_time += end_time
            warmup_time += end_time

            # warm up阶段同样的
            prob1, all_loss[0], all_preds[0], hist_preds[0] = eval_train(net1, all_loss[0], all_preds[0],
                                                                         hist_preds[0])
            prob2, all_loss[1], all_preds[1], hist_preds[1] = eval_train(net1, all_loss[1], all_preds[1],
                                                                         hist_preds[1])

            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)
            idx_view_labeled = (pred1).nonzero()[0]
            idx_view_unlabeled = (1 - pred1).nonzero()[0]
            all_idx_view_labeled[0].append(idx_view_labeled)
            all_idx_view_labeled[1].append((pred2).nonzero()[0])
            all_idx_view_unlabeled[0].append(idx_view_unlabeled)
            all_idx_view_unlabeled[1].append((1 - pred2).nonzero()[0])

            if epoch == (warm_up - 1):
                time_log.write('Warmup: %f \n' % (warmup_time))
                time_log.flush()
        # elif (epoch+1)%mid_warmup==0:
        #     lr = 0.001
        #     for param_group in optimizer1.param_groups:
        #         param_group['lr'] = lr
        #     for param_group in optimizer2.param_groups:
        #         param_group['lr'] = lr
        #
        #     warmup_trainloader = loader.run(0.5, 'warmup')
        #     print('Mid-training Warmup Net1')
        #     warmup(epoch, net1, optimizer1, warmup_trainloader)
        #     print('\nMid-training Warmup Net2')
        #     warmup(epoch, net2, optimizer2, warmup_trainloader)
        else:
            start_time = time.time()

            prob1, all_loss[0], all_preds[0], hist_preds[0] = eval_train(net1, all_loss[0], all_preds[0],
                                                                         hist_preds[0])

            prob2, all_loss[1], all_preds[1], hist_preds[1] = eval_train(net2, all_loss[1], all_preds[1],
                                                                         hist_preds[1])

            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)

            idx_view_labeled = (pred1).nonzero()[0]
            idx_view_unlabeled = (1 - pred1).nonzero()[0]
            all_idx_view_labeled[0].append(idx_view_labeled)
            all_idx_view_labeled[1].append((pred2).nonzero()[0])
            all_idx_view_unlabeled[0].append(idx_view_unlabeled)
            all_idx_view_unlabeled[1].append((1 - pred2).nonzero()[0])

            # check hist of predclean
            superclean = []
            nclean = args.num_clean
            # for ii in range(50000):
            for ii in range(len(eval_loader.dataset)):
                clean_lastn = True
                for h_ep in all_idx_view_labeled[0][-nclean:]:  # check last nclean epochs
                    if ii not in h_ep:  # 主要有一个不是true，ii就不会在里面
                        clean_lastn = False
                        break
                if clean_lastn:
                    superclean.append(ii)
            print('\nsuperclean: %d' % len(superclean))
            all_superclean[0].append(superclean)
            pred1 = np.array([True if p in superclean else False for p in range(len(pred1))])

            superclean = []
            nclean = args.num_clean
            # for ii in range(50000):
            for ii in range(len(eval_loader.dataset)):
                clean_lastn = True
                for h_ep in all_idx_view_labeled[1][-nclean:]:  # check last nclean epochs
                    if ii not in h_ep:
                        clean_lastn = False
                        break
                if clean_lastn:
                    superclean.append(ii)
            all_superclean[1].append(superclean)
            pred2 = np.array([True if p in superclean else False for p in range(len(pred2))])

            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run(0.5, 'train', pred2, prob2)  # co-divide
            train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)  # train net1

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run(0.5, 'train', pred1, prob1)  # co-divide
            train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net2
            end_time = round(time.time() - start_time)
            total_time += end_time

        save_models(epoch, net1, optimizer1, net2, optimizer2, path_exp)
        web_acc = test(epoch,net1,net2,web_valloader)
        imagenet_acc = test(epoch,net1,net2,imagenet_valloader)

        print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
        test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n'%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
        test_log.flush()

        # scheduler1.step()
        # scheduler2.step()

        if epoch ==100:
            model_name_1 = 'Net1_100epochs.pth'
            model_name_2 = 'Net2_100epochs.pth'

            print("Save the Model at 100 epochs-----")
            torch.save(net1.module.state_dict(), os.path.join(model_save_loc, model_name_1))
            torch.save(net2.module.state_dict(), os.path.join(model_save_loc, model_name_2))


        if web_acc[0] > best_acc:
            if epoch <warm_up:
                model_name_1 = 'Net1_warmup.pth'
                model_name_2 = 'Net2_warmup.pth'
            else:
                model_name_1 = 'Net1.pth'
                model_name_2 = 'Net2.pth'

            print("Save the Model-----")
            checkpoint1 = {
                'net': net1.module.state_dict(),
                'Model_number': 1,
                'Loss Function': 'CrossEntropyLoss',
                'Optimizer': 'SGD',
                'Accuracy': web_acc,
                'Dataset': 'WebVision',
                'epoch': epoch,
            }

            checkpoint2 = {
                'net': net2.module.state_dict(),
                'Model_number': 2,
                'Loss Function': 'CrossEntropyLoss',
                'Optimizer': 'SGD',
                'Accuracy': web_acc,
                'Dataset': 'WebVision',
                'epoch': epoch,
            }

            torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
            torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))
            best_acc = web_acc[0]


 

