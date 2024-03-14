from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image
from torchvision import transforms
from trainer import Continual
import sys
from torch.utils.data import Dataset
from torchvision import transforms
from types import SimpleNamespace
import yaml
import argparse
import torch
import random

import os

import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import os

class iDomainNet(Dataset):

    def __init__(self, root='../datasets/DomainNet/data',
                 data_manager=None,
                 train=False,
                 # transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):




        class_order = np.arange(345).tolist()
        # class_order = np.arange(6 * 345).tolist()
        self.class_order = class_order
        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]
        self.TestData = []
        self.TestLabels = []
        self.data_manager = data_manager


        use_path = True
        train_trsf = [
            transforms.Resize(32),
            # transforms.RandomResizedCrop(32),
            # transforms.RandomHorizontalFlip(),
        ]
        test_trsf = [
            transforms.Resize(32),
            # transforms.CenterCrop(32),
        ]
        # train_trsf = [
        #     transforms.RandomResizedCrop(224),
        #     # transforms.RandomHorizontalFlip(),
        # ]
        # test_trsf = [
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        # ]
        common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.train_transform = transforms.Compose([transforms.Resize(32),
                                                   transforms.RandomResizedCrop(32),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225]),
                                                   ])

        self.test_transform = transforms.Compose([transforms.Resize(32),
                                                  transforms.CenterCrop(32),
                                                  transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225]),
                                              ])
    def download_data(self, taskID):
        self.image_list_root = "../datasets/DomainNet/data"
        self.image_list_paths = [os.path.join(self.image_list_root, self.domain_names[taskID] + "_" + "train" + ".txt")]

        print(self.image_list_paths)
        # image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names[taskID]]
        imgs = []
        for taskid, image_list_path in enumerate(self.image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)

        self.image_test_list_paths = [
            os.path.join(self.image_list_root, self.domain_names[taskID] + "_" + "test" + ".txt")]
        imgs = []
        for taskid, image_list_path in enumerate(self.image_test_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.TestData = np.array(train_x)
        self.TestLabels = np.array(train_y)

    def getTrainData(self, label):
        datas, labels = [], []

        data = self.train_data[np.array(self.train_targets) == label]
        datas.append(data)
        labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def append_task_dataset(self, task_id, domain):
        datas, labels = [], []

        for label in self.data_manager.classes_per_task[task_id]:

            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            data = self.data[np.array(self.targets) == actual_label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))

        # print(datas, labels)
        if len(datas) > 0 and len(labels) > 0:
            datas, labels = self.concatenate(datas, labels)
            self.TestData = datas if self.TestData == [] else np.concatenate((self.TestData, datas), axis=0)
            self.TestLabels = labels if self.TestLabels == [] else np.concatenate((self.TestLabels, labels), axis=0)
            # print("the size of test set is %s"%(str(self.TestData.shape)))
            # print("the size of test label is %s"%str(self.TestLabels.shape))
        # print("test unique : ", np.unique(self.TestLabels))

    def __getitem__(self, index):

        img = self.pil_loader(self.TrainData[index])
            # img = img.convert('RGB')
        target =  self.TrainLabels[index]
        imge = self.train_transform(img)
        return imge, target

    def pil_loader(self,path):
        """
        Ref:
        https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
        """
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    def __len__(self):
        return len(self.TrainLabels)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]



def experiment(final_params):

    runs = final_params.run

    for num_run in range(runs):
        print(f"#RUN{num_run}")
        
        if num_run == 0:
            if hasattr(final_params, 'filename'):
                org_filename = final_params.filename
            else:
                org_filename = ""
        
        final_params.filename = org_filename + f'run{num_run}'

        
        if num_run == 0 and hasattr(final_params, 'rb_path'):
            org_rb_path = final_params.rb_path
            print(final_params.rb_path)
        if hasattr(final_params, 'rb_path'):
            final_params.rb_path = org_rb_path + '/' + f'{final_params.filename}'
            os.makedirs(final_params.rb_path, exist_ok=True)

            print(final_params.rb_path)
        print(final_params.filename)


        
        if hasattr(final_params, 'seed_start'):
            if final_params.seed_start is not None:
                seed = final_params.seed_start + num_run
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                print("SEED : ", seed)

                final_params.seed = seed


        if hasattr(final_params, 'result_save_path'):
            os.makedirs(final_params.result_save_path, exist_ok=True)
            print(final_params.result_save_path)

        print(final_params.filename)

        num_task = final_params.num_task_cls_per_task[0]
        num_classes_per_task = final_params.num_task_cls_per_task[1]

        class_order = np.arange(100)

        if final_params.data_order == 'seed':
            np.random.shuffle(class_order)
        # order from https://github.com/arthurdouillard/incremental_learning.pytorch/blob/master/options/data/imagenet1000_1order.yaml
        elif final_params.data_order == 'fixed':
            #class_order = [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]

            
            class_order = [0,   1,  2,  3,  4,  5,  6,  7,  8, 9,
                           10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                           20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                           30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                           40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                           50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                           60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                           70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                           80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                           90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                           100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                           ]
            
        print(class_order)

        continual = Continual(**vars(final_params))

        dataset = iDomainNet()
        # 这里建立每个任务的数据集，这里已经传过transformer了，还需要再传嘛，要注意，我觉着后面就不用再传了！
        for task_id in range(6):
            dataset.download_data(task_id)
            label_st = task_id
            for x in range(label_st * 10, label_st*10 + 60 ):
                print(class_order[x])
                dataset.getTrainData(class_order[x])

                #print(dataset.TrainLabels)

                for i in range(len(dataset)):
                    img, label = dataset[i]
                    # 这里数据集保存在了 continual类中的stream_dataset（MultiTaskStreamDataset）中
                    # 数据被发送到了  self.data_queue中
                    continual.send_stream_data(img, label, task_id)
            continual.train_disjoint(task_id,task_id)

        del continual
