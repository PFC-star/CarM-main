from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, ImageFolder

from PIL import Image
import numpy as np
import os
import pickle
import torch
import random
def get_test_set(test_set_name, data_manager, test_transform):
    print(test_set_name)
    test_set = {
        "imagenet" : ImagenetTestDataset,
        "imagenet100" : ImagenetTestDataset,
        "imagenet1000" : ImagenetTestDataset,
        "tiny_imagenet" : TinyImagenetTestDataset,
        "cifar100" : Cifar100TestDataset,
        "mini_imagenet" : MiniImagenetTestDataset,
        "cifar10" : Cifar10TestDataset,
        "domainNet": iDomainNet,
    }
    if test_set == "imagenet100":
        return ImagenetTestDataset(data_manager=data_manager, test_transform=test_transform, num_class=100)
    else:
        return test_set[test_set_name](data_manager=data_manager, test_transform=test_transform)


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
            transforms.Resize(64),
            transforms.RandomResizedCrop(64),
            # transforms.RandomHorizontalFlip(),
        ]
        test_trsf = [
            transforms.Resize(68),
            transforms.CenterCrop(64),
        ]
        # train_trsf = [
        #     transforms.RandomResizedCrop(224),
        #     # transforms.RandomHorizontalFlip(),
        # ]
        # test_trsf = [
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        # ]


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

        img = self.pil_loader(self.TestData[index])
            # img = img.convert('RGB')
        target =  self.TestLabels[index]
        imge = self.test_transform(img)
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
        return len(self.TestLabels)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]
class ImagenetTestDataset(Dataset):
    def __init__(self,
                 root='/data/Imagenet',
                 #root='/data',
                 data_manager=None,
                 split='val',
                 test_transform=None,
                 target_transform=None,
                 num_class=1000
                 ):
                 
        self.data_manager = data_manager
        self.test_transform = test_transform

        self.num_class = num_class

        if self.num_class == 1000:
            self.data_paths, self.labels = self.load_data('data/imagenet-1000/val.txt')
        elif self.num_class == 100:
            self.data_paths, self.labels = self.load_data('data/imagenet-100/val.txt')

        self.data = list()
        self.targets = list()

    def load_data(self, fpath):
        data = []
        labels = []

        lines = open(fpath)
        
        for i in range(self.num_class):
            data.append([])
            labels.append([])

        for line in lines:
            arr = line.strip().split()
            data[int(arr[1])].append(arr[0])
            labels[int(arr[1])].append(int(arr[1]))

        return data, labels

    def append_task_dataset(self, task_id,domain):
        print("data_manager.classes_per_task[task_id] : ", self.data_manager.classes_per_task[task_id])
        for label in self.data_manager.classes_per_task[task_id]:
            actual_label = self.data_manager.map_int_label_to_str_label[label]

            if label in self.targets:
                continue
            for data_path in self.data_paths[actual_label]:
                data_path = os.path.join('/data/Imagenet', data_path)
                with open(data_path,'rb') as f:
                    img = Image.open(f)
                    img = img.convert('RGB')

                self.data.append(img)
                self.targets.append(label)
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class TinyImagenetTestDataset(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str='/data', train: bool=False, 
                 data_manager=None, test_transform: transforms=None,
                 download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = '/data'
        self.train = train
        self.download = download
        self.data_manager = data_manager
        self.test_transform = test_transform

        if download:
            from google_drive_downloader import GoogleDriveDownloader as gdd
            # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
            print('Downloading dataset')
            gdd.download_file_from_google_drive(
                file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',
                dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                unzip=True)

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))

        self.TestData = []
        self.TestLabels = []
    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def append_task_dataset(self, task_id):
        datas,labels=[],[]

        for label in self.data_manager.classes_per_task[task_id]:
            
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            data = self.data[np.array(self.targets) == actual_label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        
        #print(datas, labels)
        if len(datas)>0 and len(labels)>0:
            datas,labels=self.concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
            print("the size of test set is %s"%(str(self.TestData.shape)))
            print("the size of test label is %s"%str(self.TestLabels.shape))
        print("test unique : ", np.unique(self.TestLabels))

    def __getitem__(self, index):
        img, target = Image.fromarray(np.uint8(255 *self.TestData[index])), self.TestLabels[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]


class MiniImagenetTestDataset(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self,root='/data',
                 data_manager=None,
                 train=False,
                 #transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True) -> None:
        self.root = '/data'
        self.train = train
        self.data_manager = data_manager
        self.test_transform = test_transform
        
        self.data = []
        self.targets = []

        self.TestData = []
        self.TestLabels = []

        train_in = open(root+"/mini_imagenet/mini-imagenet-cache-train.pkl", "rb")
        train = pickle.load(train_in)
        train_x = train["image_data"].reshape([64, 600, 84, 84, 3])
        val_in = open(root+"/mini_imagenet/mini-imagenet-cache-val.pkl", "rb")
        val = pickle.load(val_in)
        val_x = val['image_data'].reshape([16, 600, 84, 84, 3])
        test_in = open(root+"/mini_imagenet/mini-imagenet-cache-test.pkl", "rb")
        test = pickle.load(test_in)
        test_x = test['image_data'].reshape([20, 600, 84, 84, 3])
        all_data = np.vstack((train_x, val_x, test_x))

        TEST_SPLIT = 1 / 6

        test_data = []
        test_label = []
        for i in range(len(all_data)):
            cur_x = all_data[i]
            cur_y = np.ones((600,)) * i
            x_test = cur_x[: int(600 * TEST_SPLIT)]
            y_test = cur_y[: int(600 * TEST_SPLIT)]
            test_data.append(x_test)
            test_label.append(y_test)

        self.data = np.concatenate(test_data)
        self.targets = np.concatenate(test_label)
        self.targets = torch.from_numpy(self.targets).type(torch.LongTensor)

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def append_task_dataset(self, task_id):
        datas,labels=[],[]

        for label in self.data_manager.classes_per_task[task_id]:
            
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            data = self.data[np.array(self.targets) == actual_label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        
        #print(datas, labels)
        if len(datas)>0 and len(labels)>0:
            datas,labels=self.concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
            #print("the size of test set is %s"%(str(self.TestData.shape)))
            #print("the size of test label is %s"%str(self.TestLabels.shape))
        #print("test unique : ", np.unique(self.TestLabels))

    def __getitem__(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]

class Cifar100TestDataset(CIFAR100):
    def __init__(self,root='../datasets',
                 data_manager=None,
                 train=False,
                 #transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True):
        super().__init__(root,train=train,
                        transform=test_transform,
                        target_transform=target_transform,
                        download=download)

        self.TestData = []
        self.TestLabels = []
        self.data_manager = data_manager
        self.test_transform = test_transform

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def append_task_dataset(self, task_id,domain):
        datas,labels=[],[]

        for label in self.data_manager.classes_per_task[task_id]:
            
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            data = self.data[np.array(self.targets) == actual_label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        
        #print(datas, labels)
        if len(datas)>0 and len(labels)>0:
            datas,labels=self.concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
            #print("the size of test set is %s"%(str(self.TestData.shape)))
            #print("the size of test label is %s"%str(self.TestLabels.shape))
        #print("test unique : ", np.unique(self.TestLabels))

    def __getitem__(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]



class Cifar10TestDataset(CIFAR10):
    def __init__(self,root='../datasets',
                 data_manager=None,
                 train=False,
                 #transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=True):
        super().__init__(root,train=train,
                        transform=test_transform,
                        target_transform=target_transform,
                        download=download)

        self.TestData = []
        self.TestLabels = []
        self.data_manager = data_manager
        self.test_transform = test_transform
        print("test_transform : ", self.test_transform)

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label
    def clean_dataset(self):
        self.TestData = []
        self.TestLabels = []
    def append_task_dataset(self, task_id,domain):
        print("data_manager.classes_per_task[task_id] : ", self.data_manager.classes_per_task[task_id])
        datas,labels=[],[]

        # for label in self.data_manager.classes_per_task[task_id]:
        for label in self.data_manager.classes_per_task[task_id]:
            if label in self.TestLabels:
                continue

            actual_label = self.data_manager.map_int_label_to_str_label[label]

            data = self.data[np.array(self.targets) == actual_label]



            tempData=[]
            for i in range(len(data)):
                t_out = transforms.ToTensor()(data[i])
                data_ = transforms.ToPILImage()(t_out.float())

                # data_tensor = transforms.ToTensor()(data)
                data_numpy = np.array(data_)
                tempData.append(data_numpy)
            datas.append(tempData)
            labels.append(np.full((data.shape[0]), label))
        
        #print(datas, labels)
        if len(datas)>0 and len(labels)>0:
            datas,labels=self.concatenate(datas,labels)
            self.TestData=datas if self.TestData==[] else np.concatenate((self.TestData,datas),axis=0)
            self.TestLabels=labels if self.TestLabels==[] else np.concatenate((self.TestLabels,labels),axis=0)
            print("the size of test set is %s"%(str(self.TestData.shape)))
            print("the size of test label is %s"%str(self.TestLabels.shape))
        print("test unique : ", np.unique(self.TestLabels))

    def __getitem__(self, index):
        random_seed = 2023
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        img = self.test_transform(img)
        return img, target

    def __len__(self):
        return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]