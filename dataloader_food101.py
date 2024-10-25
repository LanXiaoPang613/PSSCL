from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os
import matplotlib
from autoaugment import CIFAR10Policy, ImageNetPolicy

def unpickle(file):
    fo = open(file, 'rb').read()
    size = 64 * 64 * 3 + 1
    for i in range(50000):
        arr = np.fromstring(fo[i * size:(i + 1) * size], dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3, 64, 64)).transpose((1, 2, 0))
    return img, lab

class food_dataset(Dataset):
    def __init__(self, root, transform, mode, pred=[], path=[], probability=[], num_class=10):

        self.root = root
        self.transform = transform
        self.mode = mode

        root_case = self.root+"/meta"
        self.train_img_file = root_case+"/train.txt"
        self.test_img_file = root_case+'/test.txt'
        # self.test_target_file = './image_list/test_targets.npy'
        class_list_path = root_case + '/classes.txt'
        img_ppp = self.root+"/images/"

        map_name2cat = dict()
        with open(class_list_path) as fp:
            for i, line in enumerate(fp):
                row = line.strip()
                map_name2cat[row] = i
        num_class = len(map_name2cat)
        # print('Num Classes: ', num_class)

        self.test_data = []
        self.test_labels = []
        # noise_file1 = './training_batch.json'
        # noise_file2 = './testing_batch.json'
        if mode == 'test':
            if os.path.exists(self.test_img_file):
                with open(self.test_img_file) as fp:
                    # lines = fp.readline()
                    for line in fp:
                        line = line.strip()
                        class_name, path = line.split("/")
                        path = path.split("\n")
                        target = map_name2cat[class_name]
                        self.test_labels.append(int(target))

                        self.test_data.append(img_ppp+line+'.jpg')
            else:
                assert 0
        else:
            train_labels = []
            train_data = []
            if os.path.exists(self.train_img_file):
                with open(self.train_img_file) as fp:
                    # lines = fp.readline()
                    for line in fp:
                        line = line.strip()
                        class_name, path = line.split("/")
                        path = path.split("\n")
                        target = map_name2cat[class_name]
                        train_labels.append(int(target))
                        train_data.append(img_ppp + line + '.jpg')
            else:
                assert 0
            train_labels = np.array(train_labels).astype(np.int64)
            train_data = np.array(train_data)
            if self.mode == "all":
                self.train_data = train_data
                self.train_labels = train_labels
            elif self.mode == "labeled":
                pred_idx = pred.nonzero()[0]
                # train_img = path
                train_img = train_data
                self.train_data = train_img[pred_idx]#[train_img[i] for i in pred_idx]
                self.probability = probability[pred_idx]
                # self.train_labels = train_labels[pred_idx]
                self.train_labels = train_labels[pred_idx]#[train_labels[i] for i in pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.train_data)))
            elif self.mode == "unlabeled":
                pred_idx = (1 - pred).nonzero()[0]
                # train_img = path
                train_img = train_data
                self.train_data = [train_img[i] for i in pred_idx]
                self.probability = probability[pred_idx]
                # self.train_labels = train_labels[pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.train_data)))
                self.train_labels = train_labels[pred_idx]#[train_labels[i] for i in pred_idx]

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_data[index]
            target = self.train_labels[index]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            #weak da
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            # strong da
            if self.transform[2] == None:
                img3 = img1
                img4 = img2
            else:
                img3 = self.transform[2](image)
                img4 = self.transform[3](image)
            return img1, img2, img3, img4, target, prob
        elif self.mode == 'unlabeled':
            img_path = self.train_data[index]
            image = Image.open(img_path).convert('RGB')
            # weak da
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            # strong da
            if self.transform[2] == None:
                img3 = img1
                img4 = img2
            else:
                img3 = self.transform[2](image)
                img4 = self.transform[3](image)
            return img1, img2, img3, img4
        elif self.mode == 'all':
            img_path = self.train_data[index]
            target = self.train_labels[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target,index
        elif self.mode == 'test':
            img_path = self.test_data[index]
            target = self.test_labels[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_data)
        else:
            return len(self.train_data)


class food_dataloader():
    def __init__(self, root='C:/Users/zhangqian/Desktop/all_datasets/food-101', batch_size=32, num_workers=0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root

        self.transform_strong_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),])

        self.transform_weak_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),
                                 (0.3113, 0.3192, 0.3214)),
        ])

        self.transforms = {
            "warmup": self.transform_weak_train,
            "unlabeled": [
                self.transform_weak_train,
                self.transform_weak_train,
                self.transform_strong_train,
                self.transform_strong_train
            ],
            "labeled": [
                self.transform_weak_train,
                self.transform_weak_train,
                self.transform_strong_train,
                self.transform_strong_train
            ],
            "test": self.transform_test,
        }

    def run(self, mode, pred=[], prob=[], paths=[]):
        if mode == 'warmup':
            warmup_dataset = food_dataset(self.root, transform=self.transforms["warmup"], mode='all')
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return warmup_loader
        elif mode == 'train':
            labeled_dataset = food_dataset(self.root, transform=self.transforms["labeled"], mode='labeled', pred=pred, path=paths,
                                             probability=prob)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            unlabeled_dataset = food_dataset(self.root, transform=self.transforms["unlabeled"], mode='unlabeled', pred=pred,path=paths,
                                               probability=prob)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return labeled_loader, unlabeled_loader
        elif mode == 'eval_train':
            eval_dataset = food_dataset(self.root, transform=self.transforms["test"], mode='all')
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers*5,
                pin_memory=True)
            return eval_loader
        elif mode == 'test':
            test_dataset = food_dataset(self.root, transform=self.transforms["test"], mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size*5,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return test_loader

# if __name__ == '__main__':
#     loader = animal_dataloader()
#     train_loader = loader.run('warmup')
#     import matplotlib.pyplot as plt
#     for batch_idx, (inputs, labels, idx, img_path) in enumerate(train_loader):
#         print(img_path[0])
#         plt.figure(dpi=300)
#         # plt.imshow(inputs[0])
#         plt.imshow(inputs[0].reshape(64, 64, 3))
#         plt.show()
#         plt.close()
#         print(inputs.shape())
#         print(idx)
#         print(labels, len(labels))
#     # print(train_loader.dataset.__len__())