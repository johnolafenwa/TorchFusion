import torch
import torch.nn as nn
from ..utils import get_batch_size

""" Base class for all metrics, subclasses should implement the __compute__ function
    Arguments:
        name:  name of the metric
"""

class Metric():
    def __init__(self,name):
        self.name = name
        self.__count = 0
        self.__sum = 0.0

    def reset(self):
        """

        :return:
        """
        self.__count = 0
        self.__sum = 0.0

    def update(self,predictions,targets,batch_first=True):
        """

        :param predictions:
        :param targets:
        :param batch_first:
        :return:
        """
        val = self.__compute__(predictions,targets)

        multiplier = 1
        if isinstance(predictions,list) or isinstance(predictions,tuple):
            multiplier *= len(predictions)

        batch_size = get_batch_size(predictions,batch_first)

        self.__sum = self.__sum + val
        self.__count = self.__count + batch_size * multiplier
    def getValue(self):
        """

        :return:
        """
        return (self.__sum.type(torch.FloatTensor)/self.__count).item()

    def __compute__(self,prediction,label):
        """

        :param prediction:
        :param label:
        :return:
        """
        raise NotImplementedError()

""" Acccuracy metric to compute topK accuracy
    Arguments:
        name:  name of the metric
        topK: the topK values to consider
"""
class Accuracy(Metric):
    def __init__(self,name="Accuracy",topK=1):
        """

        :param name:
        :param topK:
        """
        super(Accuracy,self).__init__(name)
        self.topK = topK

    def __compute__(self,prediction,label):
        """

        :param prediction:
        :param label:
        :return:
        """

        if isinstance(prediction,list) or isinstance(prediction,tuple):
            correct = None
            for preds,labels in zip(prediction,label):
                preds = preds.cpu().data
                labels = labels.cpu().data.long()
                _, pred = preds.topk(self.topK,1,True,True)
                pred = pred.t()
                if correct is None:
                    correct = pred.eq(labels.view(1,-1).expand_as(pred))[:self.topK].view(-1).float().sum(0,True)
                else:
                    correct += pred.eq(labels.view(1, -1).expand_as(pred))[:self.topK].view(-1).float().sum(0, True)

        else:

            predictions = prediction.type(torch.float32).cpu().data
            labels = label.cpu().data.long()
            _, pred = predictions.topk(self.topK, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))[:self.topK].view(-1).float().sum(0, True)

        return correct


class MeanConfidenceScore(Metric):
    def __init__(self,name="MeanConfidenceScore",topK=1,apply_softmax=True):
        """

        :param name:
        :param topK:
        :param apply_softmax:
        """
        super(MeanConfidenceScore,self).__init__(name)
        self.topK = topK
        self.apply_softmax = apply_softmax

    def __compute__(self,prediction,label):
        """

        :param prediction:
        :param label:
        :return:
        """

        if isinstance(prediction, list) or isinstance(prediction, tuple):
            sum = None
            for preds,labels in zip(prediction,label):
                labels = labels.long()
                if self.apply_softmax:
                    preds = nn.Softmax(dim=1)(preds)

                for i, pred in enumerate(preds):
                    y_score = pred[labels[i]]
                    val = y_score if y_score in pred.topk(self.topK)[0] else 0
                    if sum is None:
                        sum = val
                    else:
                        sum.add_(val)
        else:
            labels = label.long()
            if self.apply_softmax:
                prediction = nn.Softmax(dim=1)(prediction)

            sum = None

            for i, pred in enumerate(prediction):
                y_score = pred[labels[i]]
                val = y_score if y_score in pred.topk(self.topK)[0] else 0

                if sum is None:
                    sum = val
                else:
                    sum.add_(val)

        return sum

""" Mean Squared Error
    Arguments:
        name:  name of the metric
    
"""
class MSE(Metric):
    def __init__(self,name="MSE"):
        """

        :param name:
        """
        super(MSE,self).__init__(name)

    def __compute__(self,prediction,label):

        
        """

        :param prediction:
        :param label:
        :return:
        """
        if isinstance(prediction, list) or isinstance(prediction, tuple):
            sum = None

            for preds,labels in zip(prediction,label):
                if sum is None:
                    sum = torch.sum((preds - labels) ** 2)
                else:
                    sum += torch.sum((preds - labels) ** 2)

        else:
            sum = torch.sum((prediction - label) ** 2)

        return sum

class MAE(Metric):
    def __init__(self,name="MAE"):
        """

        :param name:
        """
        super(MAE,self).__init__(name)

    def __compute__(self,prediction,label):
        """

        :param prediction:
        :param label:
        :return:
        """
        if isinstance(prediction, list) or isinstance(prediction, tuple):
            sum = None

            for preds,labels in zip(prediction,label):
                if sum is None:
                    sum = torch.sum(torch.abs(preds - labels))
                else:
                    sum += torch.sum(torch.abs(preds - labels))


        else:
            sum = torch.sum(torch.abs(prediction - label))

        return sum
