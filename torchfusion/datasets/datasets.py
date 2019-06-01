from zipfile import ZipFile
import requests
import shutil
import os
import json
from io import open
from torch.utils.data import Dataset
from PIL import Image
import tarfile
import torchvision.transforms.transforms as transformations
import numpy as np
import torch
from torchvision.datasets import *
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx



def download_file(url,path,extract_path=None):
    """

    :param url:
    :param path:
    :param extract_path:
    :return:
    """

    data = requests.get(url, stream=True)
    with open(path, "wb") as file:
        shutil.copyfileobj(data.raw, file)

    del data
    if extract_path is not None:
        if path.endswith(".gz") or path.endswith(".tgz") :
            extract_tar(path,extract_path)
        else:
            extract_zip(path, extract_path)

def extract_zip(source_path,extract_path):
    """

    :param source_path:
    :param extract_path:
    :return:
    """
    extractor = ZipFile(source_path)
    extractor.extractall(extract_path)
    extractor.close()

def extract_tar(source_path,extract_path):
    """

    :param source_path:
    :param extract_path:
    :return:
    """
    with tarfile.open(source_path) as tar:
        tar.extractall(extract_path)



class ImagePool():
    def __init__(self,pool_size):
        """

        :param pool_size:
        """

        self.pool_size = pool_size
        if self.pool_size > 0:
            self.image_array = []
            self.num_imgs = 0


    def query(self,input_images):

        if isinstance(input_images,Variable):
            input_images = input_images.data

        if self.pool_size == 0:
            return input_images

        ret_images = []

        for image in input_images:
            image = image.unsqueeze(0)

            if self.num_imgs < self.pool_size:
                self.image_array.append(image)
                self.num_imgs += 1
                ret_images.append(image)

            else:
                prob = random.uniform(0,1)

                if prob > 0.5:
                    random_image_index = random.randint(0,self.pool_size - 1)
                    ret_images.append(self.image_array[random_image_index])
                    self.image_array[random_image_index] = image
                else:
                    ret_images.append(image)

            return torch.cat(ret_images,0)


"""Creates a dataset containing all images present in the paths specified in the image_paths array
      Args:
            image_paths: An array of paths, you can mix folders and files, relative and absolute paths
            transformations: A set of transformations to be applied per image
            recursive: causes the paths to be transvered recursively
            allowed_exts: an array of allowed image extensions
"""


class ImagesFromPaths(Dataset):
    def __init__(self,image_paths,transformations=None,recursive=True,allowed_exts=['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif']):

        """

        :param image_paths:
        :param transformations:
        :param recursive:
        :param allowed_exts:
        """

        super(ImagesFromPaths,self).__init__()

        assert (isinstance(image_paths,list) or isinstance(image_paths,tuple))

        self.transformations = transformations

        self.image_array = []

        for path in image_paths:

            if os.path.exists(path) == False:
                path = os.path.join(os.getcwd(),path)

            if os.path.isdir(path):

                if recursive:
                    for root, dirs, files in os.walk(path):
                        for fname in files:
                            fpath = os.path.join(root,fname)

                            if self.__get_extension(fpath) in allowed_exts:
                                self.image_array.append(fpath)
                else:
                    for fpath in os.listdir(path):
                        fpath = os.path.join(path,fpath)
                        if self.__get_extension(fpath) in allowed_exts or "." + self.__get_extension(fpath) in allowed_exts:
                            self.image_array.append(fpath)

            elif os.path.isfile(path):
                if self.__get_extension(path) in allowed_exts or "." + self.__get_extension(path) in allowed_exts:
                    self.image_array.append(path)

    def __get_extension(self,fpath):
        split = fpath.split(".")
        return split[len(split) - 1]


    def random_sample(self,batch_size):
        indexes = np.random.randint(0, self.__len__(), size=(batch_size))
        images = []
        for index in indexes:
            img = Image.open(self.image_array[index]).convert("RGB")

            if self.transformations is not None:
                img = self.transformations(img)
            images.append(img)

        return torch.stack(images)

    def __getitem__(self, index):

        img = Image.open(self.image_array[index]).convert("RGB")

        if self.transformations is not None:
            img = self.transformations(img)

        return img

    def __len__(self):
        return len(self.image_array)

class CMPFacades(Dataset):
    def __init__(self,root,source_transforms=None,target_transforms=None,set="train",download=False,reverse_mode=False):

        """

        :param root:
        :param source_transforms:
        :param target_transforms:
        :param set:
        :param download:
        :param reverse_mode:
        """

        super(CMPFacades,self).__init__()

        if set not in ["train","test","val"]:
            raise ValueError("Invalid set {}, must be train,test or val".format(set))

        self.images_ = []
        self.reverse_mode = reverse_mode

        self.source_transforms = source_transforms
        self.target_transforms = target_transforms

        path = os.path.join(root,"{}".format("facades",set))

        if os.path.exists(path) == False:
            if download:
                download_path = os.path.join(root,"facades.tar.gz")
                download_file("https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz",download_path,extract_path=root)
            else:
                raise ValueError("Facades dataset not found, set download=True to download it")
        path = os.path.join(path,set)
        for img_path in os.listdir(path):
            file_ext = self.__get_extension(img_path)
            if file_ext == "jpg":
                self.images_.append(os.path.join(path,img_path))


    def __get_extension(self, fpath):
        split = fpath.split(".")
        return split[len(split) - 1]

    def __len__(self):
        return len(self.images_)


    def __getitem__(self, index):
        img = Image.open(self.images_[index]).convert("RGB")

        if self.reverse_mode:
            img_x = img.crop((0, 0, 256, 256))
            img_y = img.crop((256, 0, 512, 256))
        else:
            img_y = img.crop((0, 0, 256, 256))
            img_x = img.crop((256, 0, 512, 256))


        if self.source_transforms is not None:
            img_x = self.source_transforms(img_x)
        if self.target_transforms is not None:
            img_y = self.target_transforms(img_y)

        return img_x,img_y

class IdenProf(ImageFolder):
    def __init__(self,root, train=True, transform=None, target_transform=None, loader=default_loader):
        """

        :param root:
        :param train:
        :param transform:
        :param target_transform:
        :param loader:
        """
        self.transform = transform
        self.target_transform = transform

        if os.path.exists(os.path.join(root,"idenprof","train","chef")) == False:
                print("Downloading {}".format("https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-jpg.zip"))
                download_file("https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-jpg.zip", "idenprof.zip", extract_path=root)

        super(IdenProf,self).__init__(root=os.path.join(root,"idenprof","train" if train else "test"),transform=transform,target_transform=target_transform,loader=loader)

def idenprof_loader(size=None,root="./idenprof",train=True,batch_size=32,mean=0.5,std=0.5,transform="default",target_transform=None,**loader_args):

    """

    :param size:
    :param root:
    :param train:
    :param batch_size:
    :param mean:
    :param std:
    :param transform:
    :param target_transform:
    :param loader_args:
    :return:
    """

    if size is not None:
        if not isinstance(size,tuple):
            size = (size,size)

    if transform == "default":
        t = []
        if size is not None:
            t.append(transformations.Resize(size))

        t.append(transformations.ToTensor())

        if mean is not None and std is not None:
            if not isinstance(mean, tuple):
                mean = (mean,)
            if not isinstance(std, tuple):
                std = (std,)
            t.append(transformations.Normalize(mean=mean, std=std))

        trans = transformations.Compose(t)


    else:
        trans = transform

    data = IdenProf(root,train=train,transform=trans,target_transform=target_transform)

    return DataLoader(data,batch_size=batch_size,shuffle=train,**loader_args)

def mnist_loader(size=None,root="./mnist",train=True,batch_size=32,mean=0.5,std=0.5,transform="default",download=True,target_transform=None,**loader_args):

    """

    :param size:
    :param root:
    :param train:
    :param batch_size:
    :param mean:
    :param std:
    :param transform:
    :param download:
    :param target_transform:
    :param loader_args:
    :return:
    """

    if size is not None:
        if not isinstance(size,tuple):
            size = (size,size)

    if transform == "default":
        t = []
        if size is not None:
            t.append(transformations.Resize(size))

        t.append(transformations.ToTensor())

        if mean is not None and std is not None:
            if not isinstance(mean, tuple):
                mean = (mean,)
            if not isinstance(std, tuple):
                std = (std,)
            t.append(transformations.Normalize(mean=mean, std=std))

        trans = transformations.Compose(t)


    else:
        trans = transform

    data = MNIST(root,train=train,transform=trans,download=download,target_transform=target_transform)

    return DataLoader(data,batch_size=batch_size,shuffle=train,**loader_args)


def cifar10_loader(size=None,root="./cifar10",train=True,batch_size=32,mean=0.5,std=0.5,transform="default",download=True,target_transform=None,**loader_args):

    """

    :param size:
    :param root:
    :param train:
    :param batch_size:
    :param mean:
    :param std:
    :param transform:
    :param download:
    :param target_transform:
    :param loader_args:
    :return:
    """

    if size is not None:
        if not isinstance(size,tuple):
            size = (size,size)

    if transform == "default":
        t = []
        if size is not None:
            t.append(transformations.Resize(size))

        t.append(transformations.ToTensor())

        if mean is not None and std is not None:
            if not isinstance(mean, tuple):
                mean = (mean,)
            if not isinstance(std, tuple):
                std = (std,)
            t.append(transformations.Normalize(mean=mean, std=std))

        trans = transformations.Compose(t)
    else:
        trans = transform

    data = CIFAR10(root,train=train,transform=trans,download=download,target_transform=target_transform)

    return DataLoader(data,batch_size=batch_size,shuffle=train,**loader_args)

def cifar100_loader(size=None,root="./cifar100",train=True,batch_size=32,mean=0.5,std=0.5,transform="default",download=True,target_transform=None,**loader_args):
    """

    :param size:
    :param root:
    :param train:
    :param batch_size:
    :param mean:
    :param std:
    :param transform:
    :param download:
    :param target_transform:
    :param loader_args:
    :return:
    """
    if size is not None:
        if not isinstance(size,tuple):
            size = (size,size)

    if transform == "default":
        t = []
        if size is not None:
            t.append(transformations.Resize(size))

        t.append(transformations.ToTensor())
        if mean is not None and std is not None:
            if not isinstance(mean, tuple):
                mean = (mean,)
            if not isinstance(std, tuple):
                std = (std,)
            t.append(transformations.Normalize(mean=mean, std=std))

        trans = transformations.Compose(t)
    else:
        trans = transform

    data = MNIST(root,train=train,transform=trans,download=download,target_transform=target_transform)

    return DataLoader(data,batch_size=batch_size,shuffle=train,**loader_args)

def fashionmnist_loader(size=None,root="./fashionmnist",train=True,batch_size=32,mean=0.5,std=0.5,transform="default",download=True,target_transform=None,**loader_args):
    """

    :param size:
    :param root:
    :param train:
    :param batch_size:
    :param mean:
    :param std:
    :param transform:
    :param download:
    :param target_transform:
    :param loader_args:
    :return:
    """

    if size is not None:
        if not isinstance(size,tuple):
            size = (size,size)

    if transform == "default":
        t = []
        if size is not None:
            t.append(transformations.Resize(size))

        t.append(transformations.ToTensor())
        if mean is not None and std is not None:
            if not isinstance(mean, tuple):
                mean = (mean,)
            if not isinstance(std, tuple):
                std = (std,)
            t.append(transformations.Normalize(mean=mean, std=std))

        trans = transformations.Compose(t)
    else:
        trans = transform

    data = FashionMNIST(root,train=train,transform=trans,download=download,target_transform=target_transform)

    return DataLoader(data,batch_size=batch_size,shuffle=train,**loader_args)

def emnist_loader(size=None,root="./emnist",train=True,batch_size=32,mean=0.5,std=0.5,transform="default",download=True,set="letters",target_transform=None,**loader_args):
    """

    :param size:
    :param root:
    :param train:
    :param batch_size:
    :param mean:
    :param std:
    :param transform:
    :param download:
    :param set:
    :param target_transform:
    :param loader_args:
    :return:
    """
    valid_sets = ('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')

    if set not in valid_sets: raise ValueError("set {}  is invalid, valid sets include {}".format(set,valid_sets))

    if size is not None:
        if not isinstance(size,tuple):
            size = (size,size)

    if transform == "default":
        t = []
        if size is not None:
            t.append(transformations.Resize(size))

        t.append(transformations.ToTensor())

        if mean is not None and std is not None:
            if not isinstance(mean, tuple):
                mean = (mean,)
            if not isinstance(std, tuple):
                std = (std,)
            t.append(transformations.Normalize(mean=mean, std=std))

        trans = transformations.Compose(t)
    else:
        trans = transform

    data = EMNIST(root,train=train,transform=trans,download=download,split=set,target_transform=target_transform)

    return DataLoader(data,batch_size=batch_size,shuffle=train,**loader_args)


def svhn_loader(size=None,root="./shvn",set="train",batch_size=32,mean=0.5,std=0.5,transform="default",download=True,target_transform=None,**loader_args):
    """

    :param size:
    :param root:
    :param set:
    :param batch_size:
    :param mean:
    :param std:
    :param transform:
    :param download:
    :param target_transform:
    :param loader_args:
    :return:
    """
    valid_sets = ('train', 'test', 'extra')

    if set not in valid_sets: raise ValueError("set {}  is invalid, valid sets include {}".format(set,valid_sets))

    if size is not None:
        if not isinstance(size,tuple):
            size = (size,size)

    if transform == "default":
        t = []
        if size is not None:
            t.append(transformations.Resize(size))

        t.append(transformations.ToTensor())

        if mean is not None and std is not None:
            if not isinstance(mean, tuple):
                mean = (mean,)
            if not isinstance(std, tuple):
                std = (std,)
            t.append(transformations.Normalize(mean=mean, std=std))

        trans = transformations.Compose(t)
    else:
        trans = transform
    data = SVHN(root,split=set,transform=trans,download=download,target_transform=target_transform)
    shuffle_mode = True if set == "train" else False
    return DataLoader(data,batch_size=batch_size,shuffle=shuffle_mode,**loader_args)

def cmpfacades_loader(size=None,root="./cmpfacades",set="train",batch_size=32,mean=0.5,std=0.5,transform="default",download=True,reverse_mode=False,**loader_args):
    """

    :param size:
    :param root:
    :param set:
    :param batch_size:
    :param mean:
    :param std:
    :param transform:
    :param download:
    :param reverse_mode:
    :param loader_args:
    :return:
    """
    valid_sets = ('train', 'test', 'val')

    if set not in valid_sets: raise ValueError("set {}  is invalid, valid sets include {}".format(set,valid_sets))

    if size is not None:
        if not isinstance(size,tuple):
            size = (size,size)

    if transform == "default":
        t = []
        if size is not None:
            t.append(transformations.Resize(size))

        t.append(transformations.ToTensor())

        if mean is not None and std is not None:
            if not isinstance(mean, tuple):
                mean = (mean,)
            if not isinstance(std, tuple):
                std = (std,)
            t.append(transformations.Normalize(mean=mean, std=std))

        trans = transformations.Compose(t)
    else:
        trans = transform
    data = CMPFacades(root,source_transforms=trans,target_transforms=trans,set=set,download=download,reverse_mode=reverse_mode)
    shuffle_mode = True if set == "train" else False
    return DataLoader(data,batch_size=batch_size,shuffle=shuffle_mode,**loader_args)


def pathimages_loader(image_paths,size=None,recursive=True,allowed_exts=['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif'],shuffle=False,batch_size=32,mean=0.5,std=0.5,transform="default",**loader_args):
    """

    :param image_paths:
    :param size:
    :param recursive:
    :param allowed_exts:
    :param shuffle:
    :param batch_size:
    :param mean:
    :param std:
    :param transform:
    :param loader_args:
    :return:
    """
    if size is not None:
        if not isinstance(size,tuple):
            size = (size,size)

    if transform == "default":
        t = []
        if size is not None:
            t.append(transformations.Resize(size))

        t.append(transformations.ToTensor())

        if mean is not None and std is not None:
            if not isinstance(mean, tuple):
                mean = (mean,)
            if not isinstance(std, tuple):
                std = (std,)
            t.append(transformations.Normalize(mean=mean, std=std))

        trans = transformations.Compose(t)
    else:
        trans = transform

    data = ImagesFromPaths(image_paths,trans,recursive=recursive,allowed_exts=allowed_exts)

    return DataLoader(data,batch_size=batch_size,shuffle=shuffle,**loader_args)

class DataFolder(DatasetFolder):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        class_map (str): a path to json mapping from class names to class index

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None,class_map=None):

        if class_map is None:
            classes, class_to_idx = find_classes(root)
        else:
            if os.path.exists(class_map):
                with open(class_map) as f:
                    c_map = json.load(f)
                classes = [c for c in c_map]
                class_to_idx = c_map
            else:
                classes, class_to_idx = find_classes(root)
                with open(class_map,"w") as f:
                    json.dump(class_to_idx,f)


        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform



def imagefolder_loader(size=None,root="./data",shuffle=False,class_map=None,batch_size=32,mean=0.5,std=0.5,transform="default",allowed_exts=['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif'],source=None,target_transform=None,**loader_args):

    """

    :param size:
    :param root:
    :param shuffle:
    :param class_map:
    :param batch_size:
    :param mean:
    :param std:
    :param transform:
    :param allowed_exts:
    :param source:
    :param target_transform:
    :param loader_args:
    :return:
    """

    if source is not None:
        if os.path.exists(root) == False:
            print("Downloading {}".format(source[0]))
            download_file(source[0],source[1],extract_path=root)
    elif len(os.listdir(root)) == 0:
        print("Downloading {}".format(source[0]))
        download_file(source[0], source[1], extract_path=root)

    if size is not None:
        if not isinstance(size,tuple):
            size = (size,size)

    if transform == "default":
        t = []
        if size is not None:
            t.append(transformations.Resize(size))

        t.append(transformations.ToTensor())

        if mean is not None and std is not None:
            if not isinstance(mean, tuple):
                mean = (mean,)
            if not isinstance(std, tuple):
                std = (std,)
            t.append(transformations.Normalize(mean=mean, std=std))

        trans = transformations.Compose(t)
    else:
        trans = transform

    data = DataFolder(root=root,loader=default_loader,extensions=allowed_exts,transform=trans,target_transform=target_transform,class_map=class_map)

    return DataLoader(data,batch_size=batch_size,shuffle=shuffle,**loader_args)




