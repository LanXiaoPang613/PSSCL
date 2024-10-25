import copy

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
from autoaugment import CIFAR10Policy


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[],
                 log='', second_ind=False):

        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise
        self.second_ind = second_ind

        self.num_samples = 50000
        self.num_classes = 10 if dataset == 'cifar10' else 100

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            self.temp_targets = np.array(train_label)
            self.temp_data = np.array(train_data)
            self.min_target = min(self.temp_targets)
            self.max_target = max(self.temp_targets)

            noise_types = "rog"
            if os.path.exists(noise_file):
                if noise_types == "plc":
                    noise_label1 = json.load(open(noise_file, "r"))['noise_labels']
                    noise_label = copy.deepcopy(train_label)
                    noise_label[:len(noise_label1)] = noise_label1
                    noise_num = len(np.where(np.array(noise_label) == np.array(train_label))[0])
                    print('plc idn noisy file found, %d.\n'%noise_num)
                elif noise_types == "pdl":
                    # part-dependent label noise, xia xiaobo
                    noise_label = json.load(open(noise_file, "r"))['noise_labels']
                    noise_num = len(np.where(np.array(noise_label) == np.array(train_label))[0])
                    print('pdl idn noisy file found, %d.\n'%noise_num)
                else:
                    # robust ... kim lee
                    noise_label = json.load(open(noise_file, "r"))
                    noise_num = len(np.where(np.array(noise_label) == np.array(train_label))[0])
                    print('RoG idn noisy file found, %.2f.\n'%(noise_num/50000.))
            else:
                if noise_types != "pdl":
                    print('error file')# inject noise
                    assert 0
                else:
                    noise_label = self.instance_noise(tau=r)
                    noise_label = np.array(noise_label).astype(np.int64)
                    print("Save pdl idn noisy labels to %s ..." % noise_file)
                    np.savez(noise_file, noise_labels=noise_label)

            noise_label = np.array(noise_label).astype(np.int64)
            train_label = np.array(train_label).astype(np.int64)
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = np.array(noise_label).astype(np.int64)
                self.train_label = np.array(train_label).astype(np.int64)
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    clean = (np.array(noise_label) == np.array(train_label))
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability, clean)
                    auc, _, _ = auc_meter.value()
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n' % (pred.sum(), auc))
                    log.flush()

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                # self.noise_label = [noise_label[i] for i in pred_idx]
                # pred_idx = np.array(pred_idx)
                self.noise_label = np.array(noise_label)[pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))

    def instance_noise(
        self,
        tau: float = 0.2,
        std: float = 0.1,
        feature_size: int = 3 * 32 * 32,
        # seed: int = 1
    ):
        '''
        Thanks the code from https://github.com/SML-Group/Label-Noise-Learning wrote by SML-Group.
        LabNoise referred much about the generation of instance-dependent label noise from this repo.
        '''
        from scipy import stats
        from math import inf
        import torch.nn.functional as F

        # np.random.seed(int(seed))
        # torch.manual_seed(int(seed))
        # torch.cuda.manual_seed(int(seed))

        # common-used parameters
        num_samples = self.num_samples
        num_classes = self.num_classes

        P = []
        # sample instance flip rates q from the truncated normal distribution N(\tau, {0.1}^2, [0, 1])
        flip_distribution = stats.truncnorm((0 - tau) / std, (1 - tau) / std,
                                            loc=tau,
                                            scale=std)
        '''
        The standard form of this distribution is a standard normal truncated to the range [a, b]
        notice that a and b are defined over the domain of the standard normal. 
        To convert clip values for a specific mean and standard deviation, use:

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        truncnorm takes  and  as shape parameters.

        so the above `flip_distribution' give a truncated standard normal distribution with mean = `tau`,
        range = [0, 1], std = `std`
        '''
        # import ipdb; ipdb.set_trace()
        # how many random variates you need to get
        q = flip_distribution.rvs(num_samples)
        # sample W \in \mathcal{R}^{S \times K} from the standard normal distribution N(0, 1^2)
        W = torch.tensor(
            np.random.randn(num_classes, feature_size,
                            num_classes)).float().cuda()  #K*dim*K, dim=3072
        for i in range(num_samples):
            x, y = self.transform(Image.fromarray(self.temp_data[i])), torch.tensor(self.temp_targets[i])
            x = x.cuda()
            # step (4). generate instance-dependent flip rates
            # 1 x feature_size  *  feature_size x 10 = 1 x 10, p is a 1 x 10 vector
            p = x.reshape(1, -1).mm(W[y]).squeeze(0)  #classes
            # step (5). control the diagonal entry of the instance-dependent transition matrix
            # As exp^{-inf} = 0, p_{y} will be 0 after softmax function.
            p[y] = -inf
            # step (6). make the sum of the off-diagonal entries of the y_i-th row to be q_i
            p = q[i] * F.softmax(p, dim=0)
            p[y] += 1 - q[i]
            P.append(p)
        P = torch.stack(P, 0).cpu().numpy()
        l = [i for i in range(self.min_target, self.max_target + 1)]
        new_label = [np.random.choice(l, p=P[i]) for i in range(num_samples)]

        print('noise rate = ', (new_label != np.array(self.temp_targets)).mean())
        new_targets = new_label
        new_targets = np.array(new_targets).astype(np.int64)
        return new_targets

    def __getitem__(self, index):
        if self.second_ind:
            if self.mode == 'labeled':
                img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
                img = Image.fromarray(img)
                img1 = self.transform[0](img)
                img2 = self.transform[1](img)
                img3 = self.transform[2](img)
                img4 = self.transform[3](img)

                return img1, img2, img3, img4, target, prob
            elif self.mode == 'unlabeled':
                img = self.train_data[index]
                img = Image.fromarray(img)
                img1 = self.transform[0](img)
                img2 = self.transform[1](img)
                img3 = self.transform[2](img)
                img4 = self.transform[3](img)
                return img1, img2, img3, img4
        else:
            if self.mode == 'labeled':
                img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
                img = Image.fromarray(img)
                img1 = self.transform(img)
                img2 = self.transform(img)
                return img1, img2, target, prob
            elif self.mode == 'unlabeled':
                img = self.train_data[index]
                img = Image.fromarray(img)
                img1 = self.transform(img)
                img2 = self.transform(img)
                return img1, img2

        if self.mode == 'all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset == 'cifar10':
            transform_weak_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            transform_strong_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_10,
                "unlabeled": [
                    transform_weak_10,
                    transform_weak_10,
                    transform_strong_10,
                    transform_strong_10
                ],
                "labeled": [
                    transform_weak_10,
                    transform_weak_10,
                    transform_strong_10,
                    transform_strong_10
                ],
            }

            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        elif self.dataset == 'cifar100':
            transform_weak_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            transform_strong_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_100,
                "unlabeled": [
                    transform_weak_100,
                    transform_weak_100,
                    transform_strong_100,
                    transform_strong_100
                ],
                "labeled": [
                    transform_weak_100,
                    transform_weak_100,
                    transform_strong_100,
                    transform_strong_100
                ],
            }
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
        self.transform_train = self.transforms

    def run(self, mode, pred=[], prob=[], second_ind=False):
        if mode == 'warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=self.transform_train["warmup"], mode="all",
                                        noise_file=self.noise_file, second_ind=second_ind)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                            root_dir=self.root_dir, transform=self.transform_train["labeled"],
                                            mode="labeled",
                                            noise_file=self.noise_file, pred=pred, probability=prob, log=self.log,
                                            second_ind=second_ind)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                              root_dir=self.root_dir, transform=self.transform_train["unlabeled"],
                                              mode="unlabeled",
                                              noise_file=self.noise_file, pred=pred, second_ind=second_ind)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_trainloader_map = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader, unlabeled_trainloader_map

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='test',
                                         second_ind=second_ind)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir,
                                         transform=self.transform_test, mode='all', noise_file=self.noise_file,
                                         second_ind=second_ind)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
