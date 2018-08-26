import torch


class ToTensor(object):

    def __call__(self, input):
        """

        :param input:
        :return:
        """
        return torch.tensor(input)

class Normalize(object):
    def __init__(self,mean,std):
        """

        :param mean:
        :param std:
        """
        self.mean = mean
        self.std = std
    def __call__(self, input):
        """

        :param input:
        :return:
        """
        return input.sub_(self.mean).div_(self.std)

class Compose(object):
    def __init__(self,transforms):
        """

        :param transforms:
        """

        self.transforms = transforms

    def __call__(self, input):
        """

        :param input:
        :return:
        """

        for trans in self.transforms:
            input = trans(input)
        return input

