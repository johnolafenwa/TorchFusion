import torch
import torch.nn as nn

""" Base class for all metrics, subclasses should implement the __compute__ function
    Arguments:
        name:  name of the metric
"""

class Metric():
    def __init__(self,name):
        self.name = name
        self.__count = 0
        self.__sum = 0.0
        self.history = []
    def add_history(self):
        self.history.append(self.getValue())

    def reset(self):
        self.__count = 0
        self.__sum = 0.0

    def update(self,prediction,target):
        val = self.__compute__(prediction,target)
        self.__sum = self.__sum + val
        self.__count = self.__count+ target.size(0)

    def getValue(self):
        return (self.__sum.type(torch.FloatTensor)/self.__count).item()


    def __compute__(self,prediction,label):
        raise NotImplementedError()

""" Acccuracy metric to compute topK accuracy
    Arguments:
        name:  name of the metric
        topK: the topK values to consider
"""
class Accuracy(Metric):
    def __init__(self,name="Accuracy",topK=1):
        super(Accuracy,self).__init__(name)
        self.topK = topK

    def __compute__(self,prediction,label):

        if self.topK == 1:
            _, pred = torch.max(prediction, self.topK)

            correct = torch.sum(pred == label)
        else:
            _, pred = prediction.topk(self.topK,1,True,True)
            pred = pred.t()
            correct = pred.eq(label.view(1,-1).expand_as(pred))[:self.topK].view(-1).float().sum(0,True)

        return correct


class MeanConfidenceScore(Metric):
    def __init__(self,name="MeanConfidenceScore",topK=1,apply_softmax=True):
        super(MeanConfidenceScore,self).__init__(name)
        self.topK = topK
        self.apply_softmax = apply_softmax

    def __compute__(self,prediction,label):
        if self.apply_softmax:
            prediction = nn.Softmax(dim=1)(prediction)
        sum = torch.FloatTensor([0.0])

        for i, pred in enumerate(prediction):
            y_score = pred[label[i]]
            val = y_score if y_score in pred.topk(self.topK)[0] else 0

            sum.add_(val)

        return sum

""" Mean Squared Error
    Arguments:
        name:  name of the metric
    
"""
class MSE(Metric):
    def __init__(self,name):
        super(MSE,self).__init__(name)

    def compute(self,prediction,label):
        return torch.sum((prediction - label) ** 2)
