from zipfile import ZipFile
import requests
import shutil
import os
import json
from io import open
from torchvision.datasets.lsun import LSUN
from torch.utils.data import Dataset
import warnings
from PIL import Image

def download_file(url,path,extract_path=None):

    data = requests.get(url, stream=True)
    with open(path, "wb") as file:
        shutil.copyfileobj(data.raw, file)

    del data
    if extract_path is not None:
        extractor = ZipFile(path)
        extractor.extractall(extract_path)
        extractor.close()
"""Creates a dataset containing all images present in the paths specified in the image_paths array
      Arguments:
            image_paths: An array of paths, you can mix folders and files, relative and absolute paths
            transformations: A set of transformations to be applied per image
            recursive: causes the paths to be transvered recursively
            allowed_exts: an array of allowed image extensions
"""

class ImagesFromPaths(Dataset):
    def __init__(self,image_paths,transformations=None,recursive=True,allowed_exts=["jpg","png","jpeg","tif"]):
        super(ImagesFromPaths).__init__()
        assert isinstance(image_paths,list)

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
                        if self.__get_extension(fpath) in allowed_exts:
                            self.image_array.append(fpath)

            elif os.path.isfile(path):
                if self.__get_extension(path) in allowed_exts:
                    self.image_array.append(path)

    def __get_extension(self,fpath):
        split = fpath.split(".")
        return split[len(split) - 1]

    def __getitem__(self, index):

        img = Image.open(self.image_array[index]).convert("RGB")
        img = self.transformations(img)

        return img

    def __len__(self):
        return len(self.image_array)
