import torch
from torch.autograd import Variable
import torch.cuda as cuda
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os
from time import time
from math import ceil
from io import open
from ..utils import PlotInput, visualize, get_model_summary,get_batch_size,clip_grads,save_model,load_model
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.onnx as onnx
import torch.backends.cudnn as cudnn

r"""Abstract base Model for training, evaluating and performing inference
All custom models should subclass this and implement train, evaluate and predict functions

    Args:
        use_cuda_if_available (boolean): If set to true, training would be done on a gpu if any is available
    
    """


class AbstractBaseLearner():
    def __init__(self, use_cuda_if_available=True):

        self.cuda = False
        self.fp16_mode = False
        if use_cuda_if_available and cuda.is_available():
            self.cuda = True
            cudnn.benchmark = True

        self.epoch_start_funcs = []
        self.batch_start_funcs = []
        self.epoch_end_funcs = []
        self.batch_end_funcs = []
        self.train_completed_funcs = []

    r"""Defines the training loop
    subclasses must override this
    """
    def train(self, *args):
        raise NotImplementedError()

    r"""Defines the evaluation loop
    subclasses must override this
    """
    def evaluate(self, *args):
        raise NotImplementedError()

    r"""Defines the validation loop
        subclasses must override this
        """
    def validate(self, *args):
        raise NotImplementedError()

    r"""Defines the prediction logic
    subclasses must override this
    """
    def predict(self, *args):
        raise NotImplementedError()

    r"""Adds a function to be called at the start of each epoch
        It should have the following signature::

            func(epoch) -> None
    """
    def half(self):
        self.fp16_mode = True

    def add_on_epoch_start(self,func):
        self.epoch_start_funcs.append(func)


    r"""Adds a function to be called at the end of each epoch
        It should have the following signature::

            func(epoch,data) -> None
        data is a dictionary containining metric values, losses and vital details at the end of the epcoh
    """
    def add_on_epoch_end(self, func):
        self.epoch_end_funcs.append(func)

    r"""Adds a function to be called at the start of each batch
        It should have the following signature::

            func(epoch,batch) -> None
    """
    def add_on_batch_start(self, func):
        self.batch_start_funcs.append(func)

    r"""Adds a function to be called at the end of each batch
        It should have the following signature::

            func(epoch,batch,data) -> None
        data is a dictionary containining metric values, losses and vital details at the end of the batch
    """
    def add_on_batch_end(self, func):
        self.batch_end_funcs.append(func)

    r"""Adds a function to be called at the end of training
        It should have the following signature::

            func(data) -> None
        data is a dictionary containining metric values, duration and vital details at the end of training
    """
    def add_on_training_completed(self, func):
        self.train_completed_funcs.append(func)

    r""" This function should return a dictionary containing information about the training including metric values.
    Child classes must override this.
        """
    def get_train_history(self):
        raise NotImplementedError()


r"""This is the base learner for training, evaluating and performing inference with a single model
All custom learners should subclass this and implement __train__, __evaluate__,__validate__ and __predict__ functions
This class already takes care of data loading, iterations and metrics, subclasses only need to define custom logic for training,
evaluation and prediction

    Args:
        model (nn.Module):  the module to be used for training, evaluation and inference.
        use_cuda_if_available (boolean): If set to true, training would be done on a gpu if any is available
"""

class BaseLearner(AbstractBaseLearner):
    def __init__(self, model, use_cuda_if_available=True):
        self.model = model
        super(BaseLearner, self).__init__(use_cuda_if_available)
        self.__train_history__ = {}
        self.train_running_loss = None
        self.train_metrics = None
        self.test_metrics = None
        self.val_metrics = None

        self.iterations = 0
        self.model_dir = os.getcwd()

    r"""Initialize model weights using pre-trained weights from the filepath

        Args:
            path (str): path to a compatible pre-defined model

        """
    def load_model(self, path):
        load_model(self.model,path)


    r"""Saves the model to the path specified
            Args:
                path (str): path to save model
                save_architecture (boolean): if True, both weights and architecture will be saved, default is False

            """
    def save_model(self, path,save_architecture=False):
        save_model(self.model,path,save_architecture)

    def train(self,*args):

        self.__train_loop__(*args)

    def __train_loop__(self, train_loader, train_metrics, test_loader=None, test_metrics=None, val_loader=None,val_metrics=None, num_epochs=10,lr_scheduler=None,
              save_models="all", model_dir=os.getcwd(),save_model_interval=1,display_metrics=True, save_metrics=True, notebook_mode=False, batch_log=True, save_logs=None,
              visdom_log=None,tensorboard_log=None, save_architecture=False):
        """

        :param train_loader:
        :param train_metrics:
        :param test_loader:
        :param test_metrics:
        :param val_loader:
        :param val_metrics:
        :param num_epochs:
        :param lr_scheduler:
        :param save_models:
        :param model_dir:
        :param save_model_interval:
        :param display_metrics:
        :param save_metrics:
        :param notebook_mode:
        :param batch_log:
        :param save_logs:
        :param visdom_log:
        :param tensorboard_log:
        :param save_architecture:
        :return:
        """

        if save_models not in ["all", "best"]:
            raise ValueError("save models must be 'all' or 'best' , {} is invalid".format(save_models))
        if save_models == "best" and test_loader is None and val_loader is None:
            raise ValueError("save models can only be best when test_loader or val_loader is provided ")

        if test_loader is not None:
            if test_metrics is None:
                raise ValueError("You must provide a metric for your test data")
            elif len(test_metrics) == 0:
                raise ValueError("test metrics cannot be an empty list")

        if val_loader is not None:
            if val_metrics is None:
                raise ValueError("You must provide a metric for your val data")
            elif len(val_metrics) == 0:
                raise ValueError("val metrics cannot be an empty list")

        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.val_metrics = val_metrics

        self.model_dir = model_dir

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        models_all = os.path.join(model_dir, "all_models")
        models_best = os.path.join(model_dir, "best_models")

        if not os.path.exists(models_all):
            os.mkdir(models_all)

        if not os.path.exists(models_best) and (test_loader is not None or val_loader is not None):
            os.mkdir(models_best)

        from tqdm import tqdm_notebook
        from tqdm import tqdm

        best_test_metric = 0.0
        best_val_metric = 0.0
        train_start_time = time()
        for e in range(num_epochs):

            print("Epoch {} of {}".format(e + 1, num_epochs))

            for metric in self.train_metrics:
                metric.reset()

            self.model.train()

            for func in self.epoch_start_funcs:
                func(e + 1)

            self.train_running_loss = torch.Tensor([0.0])
            train_loss = 0.0
            data_len = 0

            if notebook_mode and batch_log:
                progress_ = tqdm_notebook(enumerate(train_loader))
            elif batch_log:
                progress_ = tqdm(enumerate(train_loader))
            else:
                progress_ = enumerate(train_loader)

            max_batch_size = 0

            init_time = time()

            for i, data in progress_:
                for func in self.batch_start_funcs:
                    func(e + 1,i + 1)

                batch_size = get_batch_size(data)

                if max_batch_size < batch_size:
                    max_batch_size = batch_size

                self.__train_func__(data)
                self.iterations += 1
                data_len += batch_size
                train_loss = self.train_running_loss.item() / data_len

                if batch_log:
                    progress_message = ""
                    for metric in self.train_metrics:
                        progress_message += "Train {} : {}".format(metric.name, metric.getValue())
                    progress_.set_description("{}/{} batches ".format(int(ceil(data_len / max_batch_size)),
                                                                      int(ceil(len(
                                                                          train_loader.dataset) / max_batch_size))))
                    progress_dict = {"Train Loss": train_loss}

                    for metric in self.train_metrics:
                        progress_dict["Train " + metric.name] = metric.getValue()

                    progress_.set_postfix(progress_dict)
                batch_info = {"train_loss":train_loss}
                for metric in self.train_metrics:

                    metric_name = "train_{}".format(metric.name)
                    batch_info[metric_name] = metric.getValue()

                for func in self.batch_end_funcs:
                    func(e + 1,i + 1,batch_info)
            if self.cuda:
                cuda.synchronize()
            duration = time() - init_time

            if "duration" in self.__train_history__:
                self.__train_history__["duration"].append(duration)
            else:
                self.__train_history__["duration"] = [duration]


            if "train_loss" in self.__train_history__:
                self.__train_history__["train_loss"].append(train_loss)
            else:
                self.__train_history__["train_loss"] = [train_loss]

            model_file = os.path.join(models_all, "model_{}.pth".format(e + 1))

            if save_models == "all" and (e+1) % save_model_interval == 0:
                self.save_model(model_file,save_architecture)

            logfile = None
            if save_logs is not None:
                logfile = open(save_logs, "a")


            print(os.linesep + "Epoch: {}, Duration: {} , Train Loss: {}".format(e + 1, duration, train_loss))
            if logfile is not None:
                logfile.write(
                    os.linesep + "Epoch: {}, Duration: {} , Train Loss: {}".format(e + 1, duration, train_loss))

            if val_loader is None and lr_scheduler is not None:
                if isinstance(lr_scheduler,ReduceLROnPlateau):
                    lr_scheduler.step(train_metrics[0].getValue())
                else:
                    lr_scheduler.step()

            if test_loader is not None:
                message = "Test Accuracy did not improve, current best is {}".format(best_test_metric)
                current_best = best_test_metric
                self.evaluate(test_loader, test_metrics)
                result = self.test_metrics[0].getValue()

                if result > current_best:
                    best_test_metric = result
                    message = "Test {} improved from {} to {}".format(test_metrics[0].name, current_best, result)
                    model_file = os.path.join(models_best, "model_{}.pth".format(e + 1))
                    self.save_model(model_file,save_architecture)

                    print(os.linesep + "{} New Best Model saved in {}".format(message, model_file))
                    if logfile is not None:
                        logfile.write(os.linesep + "{} New Best Model saved in {}".format(message, model_file))

                else:
                    print(os.linesep + message)
                    if logfile is not None:
                        logfile.write(os.linesep + message)

                for metric in self.test_metrics:
                    metric_name = "test_{}".format(metric.name)
                    if metric_name in self.__train_history__:
                        self.__train_history__[metric_name].append(metric.getValue())
                    else:
                        self.__train_history__[metric_name] = [metric.getValue()]


                    print("Test {} : {}".format(metric.name, metric.getValue()))
                    if logfile is not None:
                        logfile.write(os.linesep + "Test {} : {}".format(metric.name, metric.getValue()))

            if val_loader is not None:
                message = "Val Accuracy did not improve, current best is {}".format(best_val_metric)
                current_best = best_val_metric
                self.validate(val_loader, val_metrics)
                result = self.val_metrics[0].getValue()

                if lr_scheduler is not None:
                    if isinstance(lr_scheduler, ReduceLROnPlateau):
                        lr_scheduler.step(result)
                    else:
                        lr_scheduler.step()

                if result > current_best:
                    best_val_metric = result
                    message = "Val {} improved from {} to {}".format(val_metrics[0].name, current_best, result)

                    if test_loader is None:
                        model_file = os.path.join(models_best, "model_{}.pth".format(e + 1))
                        self.save_model(model_file,save_architecture)

                        print(os.linesep + "{} New Best Model saved in {}".format(message, model_file))
                        if logfile is not None:
                            logfile.write(os.linesep + "{} New Best Model saved in {}".format(message, model_file))
                    else:
                        print(os.linesep + "{}".format(message))
                        if logfile is not None:
                            logfile.write(os.linesep + "{}".format(message))

                else:
                    print(os.linesep + message)
                    if logfile is not None:
                        logfile.write(os.linesep + message)

                for metric in self.val_metrics:

                    metric_name = "val_{}".format(metric.name)
                    if metric_name in self.__train_history__:
                        self.__train_history__[metric_name].append(metric.getValue())
                    else:
                        self.__train_history__[metric_name] = [metric.getValue()]

                    print("Val {} : {}".format(metric.name, metric.getValue()))
                    if logfile is not None:
                        logfile.write(os.linesep + "Val {} : {}".format(metric.name, metric.getValue()))


            for metric in self.train_metrics:

                metric_name = "train_{}".format(metric.name)
                if metric_name in self.__train_history__:
                    self.__train_history__[metric_name].append(metric.getValue())
                else:
                    self.__train_history__[metric_name] = [metric.getValue()]

                print("Train {} : {}".format(metric.name, metric.getValue()))
                if logfile is not None:
                    logfile.write(os.linesep + "Train {} : {}".format(metric.name, metric.getValue()))

            if logfile is not None:
                logfile.close()


            if "epoch" in self.__train_history__:
                self.__train_history__["epoch"].append(e+1)
            else:
                self.__train_history__["epoch"] = [e+1]
            epoch_arr = self.__train_history__["epoch"]
            epoch_arr_tensor = torch.LongTensor(epoch_arr)

            if visdom_log is not None:
                visdom_log.plot_line(torch.FloatTensor(self.__train_history__["train_loss"]),epoch_arr_tensor,win="train_loss",title="Train Loss")

                if test_metrics is not None:
                     for metric in test_metrics:
                         metric_name = "test_{}".format(metric.name)
                         visdom_log.plot_line(torch.FloatTensor(self.__train_history__[metric_name]),epoch_arr_tensor,win="test_{}".format(metric.name),title="Test {}".format(metric.name))
                if val_metrics is not None:
                     for metric in val_metrics:
                         metric_name = "val_{}".format(metric.name)
                         visdom_log.plot_line(torch.FloatTensor(self.__train_history__[metric_name]),epoch_arr_tensor,win="val_{}".format(metric.name),title="Val {}".format(metric.name))

                for metric in train_metrics:
                    metric_name = "train_{}".format(metric.name)
                    visdom_log.plot_line(torch.FloatTensor(self.__train_history__[metric_name]), epoch_arr_tensor,
                                         win="train_{}".format(metric.name), title="Train {}".format(metric.name))


            if tensorboard_log is not None:
                writer = SummaryWriter(os.path.join(model_dir,tensorboard_log))
                writer.add_scalar("logs/train_loss", train_loss, global_step=e+1)

                if test_metrics is not None:
                     for metric in test_metrics:
                         writer.add_scalar("logs/test_metrics/{}".format(metric.name), metric.getValue(),
                                           global_step=e+1)
                if val_metrics is not None:
                     for metric in val_metrics:
                         writer.add_scalar("logs/val_metrics/{}".format(metric.name), metric.getValue(),
                                           global_step=e+1)
                for metric in train_metrics:
                    writer.add_scalar("logs/train_metrics/{}".format(metric.name), metric.getValue(),
                                      global_step=e + 1)

                writer.close()

            if display_metrics or save_metrics:

                save_path = None

                if save_metrics:
                    save_path = os.path.join(model_dir, "epoch_{}_loss.png".format(e + 1))
                visualize(epoch_arr, [PlotInput(value=self.__train_history__["train_loss"], name="Train Loss", color="red")],
                          display=display_metrics,
                          save_path=save_path)

            if test_loader is not None and (display_metrics or save_metrics):
                for metric in self.test_metrics:
                    metric_name = "test_{}".format(metric.name)

                    save_path = None

                    if save_metrics:
                        save_path = os.path.join(model_dir, "test_{}_epoch_{}.png".format(metric.name, e + 1))
                    visualize(epoch_arr, [PlotInput(value=self.__train_history__[metric_name], name="Test " + metric.name, color="blue")],
                              display=display_metrics,
                              save_path=save_path)

            if val_loader is not None and (display_metrics or save_metrics):
                for metric in self.val_metrics:
                    metric_name = "val_{}".format(metric.name)

                    save_path = None

                    if save_metrics:
                        save_path = os.path.join(model_dir, "val_{}_epoch_{}.png".format(metric.name, e + 1))
                    visualize(epoch_arr, [PlotInput(value=self.__train_history__[metric_name], name="Val " + metric.name, color="blue")],
                              display=display_metrics,
                              save_path=save_path)

            for metric in self.train_metrics:
                metric_name = "train_{}".format(metric.name)
                save_path = None
                if save_metrics:
                    save_path = os.path.join(model_dir, "train_{}_epoch_{}.png".format(metric.name, e + 1))
                visualize(epoch_arr, [PlotInput(value=self.__train_history__[metric_name], name="Train " + metric.name, color="blue")],
                              display=display_metrics,
                              save_path=save_path)
            epoch_info = {"train_loss": train_loss,"duration":duration}
            for metric in self.train_metrics:
                metric_name = "train_{}".format(metric.name)
                epoch_info[metric_name] = metric.getValue()
            if self.test_metrics != None and test_loader != None:
                for metric in self.test_metrics:

                    metric_name = "test_{}".format(metric.name)
                    epoch_info[metric_name] = metric.getValue()

            if self.val_metrics != None and val_loader != None:
                for metric in self.val_metrics:

                    metric_name = "val_{}".format(metric.name)
                    epoch_info[metric_name] = metric.getValue()

            for func in self.epoch_end_funcs:
                func(e + 1,epoch_info)

        train_end_time = time() - train_start_time
        train_info = {"train_duration":train_end_time}
        for metric in self.train_metrics:
            metric_name = "train_{}".format(metric.name)
            train_info[metric_name] = metric.getValue()

        if self.test_metrics != None and test_loader != None:
            for metric in self.test_metrics:
                metric_name = "test_{}".format(metric.name)
                train_info[metric_name] = metric.getValue()


        if val_loader != None:
            for metric in self.val_metrics:
                metric_name = "train_{}".format(metric.name)
                train_info[metric_name] = metric.getValue()

        for func in self.train_completed_funcs:
            func(train_info)



    r"""Training logic, all models must override this
            Args:
                data: a single batch of data from the train_loader
    """

    def __train_func__(self, data):
        raise NotImplementedError()

    r"""Evaluates the dataset on the set of provided metrics
            Args:
                test_loader (DataLoader): an instance of DataLoader containing the test set
                test_metrics ([]): an array of metrics for evaluating the test set
        """
    def evaluate(self, test_loader, metrics):

        if self.test_metrics is None:
            self.test_metrics = metrics

        for metric in self.test_metrics:
            metric.reset()

        self.model.eval()

        for i, data in enumerate(test_loader):
            self.__eval_function__(data)


    r"""Evaluation logic, all models must override this
            Args:
                data: a single batch of data from the test_loader
        """
    def __eval_function__(self, data):
        raise NotImplementedError()

    r"""Validates the dataset on the set of provided metrics
                Args:
                    val_loader (DataLoader): an instance of DataLoader containing the test set
                    metrics ([]): an array of metrics for evaluating the test set
            """

    def validate(self, val_loader, metrics):

        if self.val_metrics is None:
            self.val_metrics = metrics

        for metric in self.val_metrics:
            metric.reset()

        self.model.eval()

        for i, data in enumerate(val_loader):
            self.__val_function__(data)


    r"""Validation logic, all models must override this
            Args:
                data: a single batch of data from the test_loader
        """

    def __val_function__(self, data):
        raise NotImplementedError()

    r"""Runs inference on the given input
            Args:
                inputs: a DataLoader or a tensor of input values.
    """
    def predict(self, inputs):
        self.model.eval()

        if isinstance(inputs, DataLoader):
            predictions = []
            for i, data in enumerate(inputs):
                batch_pred = self.__predict_func__(data)

                for pred in batch_pred:
                    predictions.append(pred.unsqueeze(0))

            return torch.cat(predictions)
        else:
            pred = self.__predict_func__(inputs)
            return pred.squeeze(0)

    r"""Inference logic, all models must override this
            Args:
                input: a batch of data 
    """

    def __predict_func__(self, input):
        raise NotImplementedError()

    r""" Returns a dictionary containing the values of metrics, epochs and loss during training.
    """
    def get_train_history(self):
        return self.__train_history__


class BaseTextLearner(BaseLearner):
    def __init__(self, model, source_field, target_field, batch_first=False,use_cuda_if_available=True):

        super(BaseTextLearner, self).__init__(model,use_cuda_if_available)

        self.batch_first = batch_first
        self.source_field = source_field
        self.target_field = target_field

    def __train_loop__(self, train_loader, train_metrics, test_loader=None, test_metrics=None, val_loader=None,val_metrics=None, num_epochs=10,lr_scheduler=None,
              save_models="all", model_dir=os.getcwd(),save_model_interval=1,display_metrics=True, save_metrics=True, notebook_mode=False, batch_log=True, save_logs=None,
              visdom_log=None,tensorboard_log=None, save_architecture=False):
        """

        :param train_loader:
        :param train_metrics:
        :param test_loader:
        :param test_metrics:
        :param val_loader:
        :param val_metrics:
        :param num_epochs:
        :param lr_scheduler:
        :param save_models:
        :param model_dir:
        :param save_model_interval:
        :param display_metrics:
        :param save_metrics:
        :param notebook_mode:
        :param batch_log:
        :param save_logs:
        :param visdom_log:
        :param tensorboard_log:
        :param save_architecture:
        :return:
        """

        if save_models not in ["all", "best"]:
            raise ValueError("save models must be 'all' or 'best' , {} is invalid".format(save_models))
        if save_models == "best" and test_loader is None and val_loader is None:
            raise ValueError("save models can only be best when test_loader or val_loader is provided ")

        if test_loader is not None:
            if test_metrics is None:
                raise ValueError("You must provide a metric for your test data")
            elif len(test_metrics) == 0:
                raise ValueError("test metrics cannot be an empty list")

        if val_loader is not None:
            if val_metrics is None:
                raise ValueError("You must provide a metric for your val data")
            elif len(val_metrics) == 0:
                raise ValueError("val metrics cannot be an empty list")

        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.val_metrics = val_metrics

        self.model_dir = model_dir

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        models_all = os.path.join(model_dir, "all_models")
        models_best = os.path.join(model_dir, "best_models")

        if not os.path.exists(models_all):
            os.mkdir(models_all)

        if not os.path.exists(models_best) and (test_loader is not None or val_loader is not None):
            os.mkdir(models_best)

        from tqdm import tqdm_notebook
        from tqdm import tqdm

        best_test_metric = 0.0
        best_val_metric = 0.0
        train_start_time = time()
        for e in range(num_epochs):

            print("Epoch {} of {}".format(e + 1, num_epochs))

            for metric in self.train_metrics:
                metric.reset()

            self.model.train()

            for func in self.epoch_start_funcs:
                func(e + 1)

            self.train_running_loss = torch.Tensor([0.0])
            train_loss = 0.0
            data_len = 0

            if notebook_mode and batch_log:
                progress_ = tqdm_notebook(enumerate(train_loader))
            elif batch_log:
                progress_ = tqdm(enumerate(train_loader))
            else:
                progress_ = enumerate(train_loader)

            max_batch_size = 0

            init_time = time()

            for i, data in progress_:
                for func in self.batch_start_funcs:
                    func(e + 1,i + 1)

                source = getattr(data, self.source_field)

                batch_size = get_batch_size(source, self.batch_first)

                if max_batch_size < batch_size:
                    max_batch_size = batch_size

                self.__train_func__(data)
                self.iterations += 1
                data_len += batch_size
                train_loss = self.train_running_loss.item() / data_len

                if batch_log:
                    progress_message = ""
                    for metric in self.train_metrics:
                        progress_message += "Train {} : {}".format(metric.name, metric.getValue())
                    progress_.set_description("{}/{} batches ".format(int(ceil(data_len / max_batch_size)),
                                                                      int(ceil(len(
                                                                          train_loader.dataset) / max_batch_size))))
                    progress_dict = {"Train Loss": train_loss}

                    for metric in self.train_metrics:
                        progress_dict["Train " + metric.name] = metric.getValue()

                    progress_.set_postfix(progress_dict)
                batch_info = {"train_loss":train_loss}
                for metric in self.train_metrics:

                    metric_name = "train_{}".format(metric.name)
                    batch_info[metric_name] = metric.getValue()

                for func in self.batch_end_funcs:
                    func(e + 1,i + 1,batch_info)
            if self.cuda:
                cuda.synchronize()
            duration = time() - init_time

            if "duration" in self.__train_history__:
                self.__train_history__["duration"].append(duration)
            else:
                self.__train_history__["duration"] = [duration]


            if "train_loss" in self.__train_history__:
                self.__train_history__["train_loss"].append(train_loss)
            else:
                self.__train_history__["train_loss"] = [train_loss]

            model_file = os.path.join(models_all, "model_{}.pth".format(e + 1))

            if save_models == "all" and (e+1) % save_model_interval == 0:
                self.save_model(model_file,save_architecture)

            logfile = None
            if save_logs is not None:
                logfile = open(save_logs, "a")


            print(os.linesep + "Epoch: {}, Duration: {} , Train Loss: {}".format(e + 1, duration, train_loss))
            if logfile is not None:
                logfile.write(
                    os.linesep + "Epoch: {}, Duration: {} , Train Loss: {}".format(e + 1, duration, train_loss))

            if val_loader is None and lr_scheduler is not None:
                if isinstance(lr_scheduler,ReduceLROnPlateau):
                    lr_scheduler.step(train_metrics[0].getValue())
                else:
                    lr_scheduler.step()

            if test_loader is not None:
                message = "Test Accuracy did not improve, current best is {}".format(best_test_metric)
                current_best = best_test_metric
                self.evaluate(test_loader, test_metrics)
                result = self.test_metrics[0].getValue()

                if result > current_best:
                    best_test_metric = result
                    message = "Test {} improved from {} to {}".format(test_metrics[0].name, current_best, result)
                    model_file = os.path.join(models_best, "model_{}.pth".format(e + 1))
                    self.save_model(model_file,save_architecture)

                    print(os.linesep + "{} New Best Model saved in {}".format(message, model_file))
                    if logfile is not None:
                        logfile.write(os.linesep + "{} New Best Model saved in {}".format(message, model_file))

                else:
                    print(os.linesep + message)
                    if logfile is not None:
                        logfile.write(os.linesep + message)

                for metric in self.test_metrics:
                    metric_name = "test_{}".format(metric.name)
                    if metric_name in self.__train_history__:
                        self.__train_history__[metric_name].append(metric.getValue())
                    else:
                        self.__train_history__[metric_name] = [metric.getValue()]


                    print("Test {} : {}".format(metric.name, metric.getValue()))
                    if logfile is not None:
                        logfile.write(os.linesep + "Test {} : {}".format(metric.name, metric.getValue()))

            if val_loader is not None:
                message = "Val Accuracy did not improve, current best is {}".format(best_val_metric)
                current_best = best_val_metric
                self.validate(val_loader, val_metrics)
                result = self.val_metrics[0].getValue()

                if lr_scheduler is not None:
                    if isinstance(lr_scheduler, ReduceLROnPlateau):
                        lr_scheduler.step(result)
                    else:
                        lr_scheduler.step()

                if result > current_best:
                    best_val_metric = result
                    message = "Val {} improved from {} to {}".format(val_metrics[0].name, current_best, result)

                    if test_loader is None:
                        model_file = os.path.join(models_best, "model_{}.pth".format(e + 1))
                        self.save_model(model_file,save_architecture)

                        print(os.linesep + "{} New Best Model saved in {}".format(message, model_file))
                        if logfile is not None:
                            logfile.write(os.linesep + "{} New Best Model saved in {}".format(message, model_file))
                    else:
                        print(os.linesep + "{}".format(message))
                        if logfile is not None:
                            logfile.write(os.linesep + "{}".format(message))

                else:
                    print(os.linesep + message)
                    if logfile is not None:
                        logfile.write(os.linesep + message)

                for metric in self.val_metrics:

                    metric_name = "val_{}".format(metric.name)
                    if metric_name in self.__train_history__:
                        self.__train_history__[metric_name].append(metric.getValue())
                    else:
                        self.__train_history__[metric_name] = [metric.getValue()]

                    print("Val {} : {}".format(metric.name, metric.getValue()))
                    if logfile is not None:
                        logfile.write(os.linesep + "Val {} : {}".format(metric.name, metric.getValue()))


            for metric in self.train_metrics:

                metric_name = "train_{}".format(metric.name)
                if metric_name in self.__train_history__:
                    self.__train_history__[metric_name].append(metric.getValue())
                else:
                    self.__train_history__[metric_name] = [metric.getValue()]

                print("Train {} : {}".format(metric.name, metric.getValue()))
                if logfile is not None:
                    logfile.write(os.linesep + "Train {} : {}".format(metric.name, metric.getValue()))

            if logfile is not None:
                logfile.close()


            if "epoch" in self.__train_history__:
                self.__train_history__["epoch"].append(e+1)
            else:
                self.__train_history__["epoch"] = [e+1]
            epoch_arr = self.__train_history__["epoch"]
            epoch_arr_tensor = torch.LongTensor(epoch_arr)

            if visdom_log is not None:
                visdom_log.plot_line(torch.FloatTensor(self.__train_history__["train_loss"]),epoch_arr_tensor,win="train_loss",title="Train Loss")

                if test_metrics is not None:
                     for metric in test_metrics:
                         metric_name = "test_{}".format(metric.name)
                         visdom_log.plot_line(torch.FloatTensor(self.__train_history__[metric_name]),epoch_arr_tensor,win="test_{}".format(metric.name),title="Test {}".format(metric.name))
                if val_metrics is not None:
                     for metric in val_metrics:
                         metric_name = "val_{}".format(metric.name)
                         visdom_log.plot_line(torch.FloatTensor(self.__train_history__[metric_name]),epoch_arr_tensor,win="val_{}".format(metric.name),title="Val {}".format(metric.name))

                for metric in train_metrics:
                    metric_name = "train_{}".format(metric.name)
                    visdom_log.plot_line(torch.FloatTensor(self.__train_history__[metric_name]), epoch_arr_tensor,
                                         win="train_{}".format(metric.name), title="Train {}".format(metric.name))


            if tensorboard_log is not None:
                writer = SummaryWriter(os.path.join(model_dir,tensorboard_log))
                writer.add_scalar("logs/train_loss",train_loss,global_step=e+1)

                if test_metrics is not None:
                     for metric in test_metrics:
                         writer.add_scalar("logs/test_metrics/{}".format(metric.name), metric.getValue(),
                                           global_step=e+1)
                if val_metrics is not None:
                     for metric in val_metrics:
                         writer.add_scalar("logs/val_metrics/{}".format(metric.name), metric.getValue(),
                                           global_step=e+1)
                for metric in train_metrics:
                    writer.add_scalar("logs/train_metrics/{}".format(metric.name), metric.getValue(),
                                      global_step=e + 1)

                writer.close()

            if display_metrics or save_metrics:

                save_path = None

                if save_metrics:
                    save_path = os.path.join(model_dir, "epoch_{}_loss.png".format(e + 1))
                visualize(epoch_arr, [PlotInput(value=self.__train_history__["train_loss"], name="Train Loss", color="red")],
                          display=display_metrics,
                          save_path=save_path)

            if test_loader is not None and (display_metrics or save_metrics):
                for metric in self.test_metrics:
                    metric_name = "test_{}".format(metric.name)

                    save_path = None

                    if save_metrics:
                        save_path = os.path.join(model_dir, "test_{}_epoch_{}.png".format(metric.name, e + 1))
                    visualize(epoch_arr, [PlotInput(value=self.__train_history__[metric_name], name="Test " + metric.name, color="blue")],
                              display=display_metrics,
                              save_path=save_path)

            if val_loader is not None and (display_metrics or save_metrics):
                for metric in self.val_metrics:
                    metric_name = "val_{}".format(metric.name)

                    save_path = None

                    if save_metrics:
                        save_path = os.path.join(model_dir, "val_{}_epoch_{}.png".format(metric.name, e + 1))
                    visualize(epoch_arr, [PlotInput(value=self.__train_history__[metric_name], name="Val " + metric.name, color="blue")],
                              display=display_metrics,
                              save_path=save_path)

            for metric in self.train_metrics:
                metric_name = "train_{}".format(metric.name)
                save_path = None
                if save_metrics:
                    save_path = os.path.join(model_dir, "train_{}_epoch_{}.png".format(metric.name, e + 1))
                visualize(epoch_arr, [PlotInput(value=self.__train_history__[metric_name], name="Train " + metric.name, color="blue")],
                              display=display_metrics,
                              save_path=save_path)
            epoch_info = {"train_loss": train_loss,"duration":duration}
            for metric in self.train_metrics:
                metric_name = "train_{}".format(metric.name)
                epoch_info[metric_name] = metric.getValue()
            if self.test_metrics != None and test_loader != None:
                for metric in self.test_metrics:

                    metric_name = "test_{}".format(metric.name)
                    epoch_info[metric_name] = metric.getValue()

            if self.val_metrics != None and val_loader != None:
                for metric in self.val_metrics:

                    metric_name = "val_{}".format(metric.name)
                    epoch_info[metric_name] = metric.getValue()

            for func in self.epoch_end_funcs:
                func(e + 1,epoch_info)

        train_end_time = time() - train_start_time
        train_info = {"train_duration":train_end_time}
        for metric in self.train_metrics:
            metric_name = "train_{}".format(metric.name)
            train_info[metric_name] = metric.getValue()

        if self.test_metrics != None and test_loader != None:
            for metric in self.test_metrics:
                metric_name = "test_{}".format(metric.name)
                train_info[metric_name] = metric.getValue()


        if val_loader != None:
            for metric in self.val_metrics:
                metric_name = "train_{}".format(metric.name)
                train_info[metric_name] = metric.getValue()

        for func in self.train_completed_funcs:
            func(train_info)




class StandardLearner(BaseLearner):
    def __init__(self, model, use_cuda_if_available=True):
        super(StandardLearner,self).__init__(model, use_cuda_if_available)

    """Train function

               Args:
                   train_loader (DataLoader): an instance of DataLoader containing the training set
                   loss_fn (Loss): the loss function 
                   optimizer (Optimizer): an optimizer for updating parameters 
                   train_metrics ([]): an array of metrics for evaluating the training set
                   test_loader (DataLoader): an instance of DataLoader containing the test set
                   test_metrics ([]): an array of metrics for evaluating the test set
                   num_epochs (int): The maximum number of training epochs
                   lr_scheduler (_LRSchedular): Learning rate scheduler updated at every epoch
                   save_models (str): If all, the model is saved at the end of each epoch while the best models based 
                       on the test set are also saved in best_models folder
                       If 'best', only the best models are saved,  test_loader and test_metrics must be provided
                   model_dir (str) : a path in which to save the models
                   save_model_interval (int): saves the models after every n epoch
                   notebook_mode (boolean): Optimizes the progress bar for either jupyter notebooks or consoles
                   display_metrics (boolean): Enables display of metrics and loss visualizations at the end of each epoch.
                   save_metrics (boolean): Enables saving of metrics and loss visualizations at the end of each epoch.
                   batch_log (boolean): Enables printing of logs at every batch iteration
                   save_logs (str): Specifies a filepath in which to permanently save logs at every epoch
                   visdom_log (VisdomLogger): Logs outputs and metrics to the visdom server
                   tensorboard_log (str): Logs outputs and metrics to the filepath for visualization in tensorboard
                   save_architecture (boolean): Saves the architecture as well as weights during model saving
                   clip_grads: a tuple specifying the minimum and maximum gradient values

                   """
    def train(self, train_loader, loss_fn, optimizer, train_metrics, test_loader=None, test_metrics=None, val_loader=None,val_metrics=None, num_epochs=10,lr_scheduler=None,
              save_models="all", model_dir=os.getcwd(),save_model_interval=1,display_metrics=False, save_metrics=False, notebook_mode=False, batch_log=True, save_logs=None,
              visdom_log=None,tensorboard_log=None, save_architecture=False,clip_grads=None):

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.clip_grads = clip_grads
        
        super().__train_loop__(train_loader, train_metrics, test_loader, test_metrics, val_loader,val_metrics, num_epochs,lr_scheduler,
              save_models, model_dir,save_model_interval,display_metrics, save_metrics, notebook_mode, batch_log, save_logs,
              visdom_log,tensorboard_log, save_architecture)

    def __train_func__(self, data):

        self.optimizer.zero_grad()

        if self.clip_grads is not None:

            clip_grads(self.model,self.clip_grads[0],self.clip_grads[1])

        train_x, train_y = data

        batch_size = get_batch_size(train_x)

        if isinstance(train_x,list) or isinstance(train_x,tuple):
            train_x = (Variable(x.cuda() if self.cuda else x) for x in train_x)
        else:
            train_x = Variable(train_x.cuda() if self.cuda else train_x)

        if isinstance(train_y,list) or isinstance(train_y,tuple):
            train_y = (Variable(y.cuda() if self.cuda else y) for y in train_y)
        else:
            train_y = Variable(train_y.cuda() if self.cuda else train_y)

        outputs = self.model(train_x)
        loss = self.loss_fn(outputs, train_y)
        if self.fp16_mode:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        self.optimizer.step()
        self.train_running_loss = self.train_running_loss + (loss.cpu().item() * batch_size)

        for metric in self.train_metrics:
            metric.update(outputs, train_y)

    def __eval_function__(self, data):

        test_x, test_y = data

        if isinstance(test_x, list) or isinstance(test_x, tuple):
            test_x = (x.cuda() if self.cuda else x for x in test_x)
        else:
            test_x = test_x.cuda() if self.cuda else test_x

        if isinstance(test_y, list) or isinstance(test_y, tuple):
            test_y = (y.cuda() if self.cuda else y for y in test_y)
        else:
            test_y = test_y.cuda() if self.cuda else test_y

        outputs = self.model(test_x)

        for metric in self.test_metrics:
            metric.update(outputs, test_y)

    def __val_function__(self, data):

        val_x, val_y = data
        if isinstance(val_x, list) or isinstance(val_x, tuple):
            val_x = (x.cuda() if self.cuda else x for x in val_x)
        else:
            val_x = val_x.cuda() if self.cuda else val_x

        if isinstance(val_y, list) or isinstance(val_y, tuple):
            val_y = (y.cuda() if self.cuda else y for y in val_y)
        else:
            val_y = val_y.cuda() if self.cuda else val_y

        outputs = self.model(val_x)

        for metric in self.val_metrics:
            metric.update(outputs, val_y)

    def __predict_func__(self, inputs):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            inputs = (x.cuda() if self.cuda else x for x in inputs)
        else:
            inputs = inputs.cuda() if self.cuda else inputs

        return self.model(inputs)

    """returns a complete summary of the model

                   Args:
                       input_sizes: a single tuple or a list of tuples in the case of multiple inputs, specifying the
                       size of the inputs to the model
                       input_types: a single tensor type or a list of tensors in the case of multiple inputs, specifying the 
                       type of the inputs to the model
                       item_length(int): the length of each item in the summary
                       tensorboard_log(str): if enabled, the model will be serialized into a format readable by tensorboard,
                       useful for visualizing the model in tensorboard.
                       """

    def summary(self,input_sizes,input_types=torch.FloatTensor,item_length=26,tensorboard_log=None):

        if isinstance(input_sizes,list):
            inputs = (torch.randn(input_size).type(input_type).unsqueeze(0) for input_size, input_type in zip(input_sizes,input_types))

            inputs = (input.cuda() if self.cuda else input for input in inputs)
        else:
            inputs = torch.randn(input_sizes).type(input_types).unsqueeze(0)

            inputs = inputs.cuda() if self.cuda else inputs


        return get_model_summary(self.model,inputs,item_length=item_length,tensorboard_log=tensorboard_log)

    """saves the model in onnx format

                       Args:
                           input_sizes: a single tuple or a list of tuples in the case of multiple inputs, specifying the
                           size of the inputs to the model
                           input_types: a single tensor type or a list of tensors in the case of multiple inputs, specifying the 
                           type of the inputs to the model
                         """
    def to_onnx(self,input_sizes,path,input_types=torch.FloatTensor,**kwargs):
        if isinstance(input_sizes,list):
            inputs = (torch.randn(input_size).type(input_type).unsqueeze(0) for input_size, input_type in zip(input_sizes,input_types))

            inputs = (input.cuda() if self.cuda else input for input in inputs)
        else:
            inputs = torch.randn(input_sizes).type(input_types).unsqueeze(0)

            inputs = inputs.cuda() if self.cuda else inputs

        return onnx._export(self.model, inputs, f=path, **kwargs)


class TextClassifier(BaseTextLearner):
    def __init__(self, model, source_field, target_field, batch_first=False, use_cuda_if_available=True):
        super(TextClassifier, self).__init__(model, source_field, target_field, batch_first, use_cuda_if_available)

    """Train function

                   Args:
                       train_loader (DataLoader): an instance of DataLoader containing the training set
                       loss_fn (Loss): the loss function 
                       optimizer (Optimizer): an optimizer for updating parameters 
                       train_metrics ([]): an array of metrics for evaluating the training set
                       test_loader (DataLoader): an instance of DataLoader containing the test set
                       test_metrics ([]): an array of metrics for evaluating the test set
                       num_epochs (int): The maximum number of training epochs
                       lr_scheduler (_LRSchedular): Learning rate scheduler updated at every epoch
                       save_models (str): If all, the model is saved at the end of each epoch while the best models based 
                           on the test set are also saved in best_models folder
                           If 'best', only the best models are saved,  test_loader and test_metrics must be provided
                       model_dir (str) : a path in which to save the models
                       save_model_interval (int): saves the models after every n epoch
                       notebook_mode (boolean): Optimizes the progress bar for either jupyter notebooks or consoles
                       display_metrics (boolean): Enables display of metrics and loss visualizations at the end of each epoch.
                       save_metrics (boolean): Enables saving of metrics and loss visualizations at the end of each epoch.
                       batch_log (boolean): Enables printing of logs at every batch iteration
                       save_logs (str): Specifies a filepath in which to permanently save logs at every epoch
                       visdom_log (VisdomLogger): Logs outputs and metrics to the visdom server
                       tensorboard_log (str): Logs outputs and metrics to the filepath for visualization in tensorboard
                       save_architecture (boolean): Saves the architecture as well as weights during model saving
                       clip_grads: a tuple specifying the minimum and maximum gradient values

                       """
    def train(self, train_loader, loss_fn, optimizer, train_metrics, test_loader=None, test_metrics=None, val_loader=None,val_metrics=None, num_epochs=10,lr_scheduler=None,
              save_models="all", model_dir=os.getcwd(),save_model_interval=1,display_metrics=False, save_metrics=False, notebook_mode=False, batch_log=True, save_logs=None,
              visdom_log=None,tensorboard_log=None, save_architecture=False,clip_grads=None):

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.clip_grads = clip_grads
        
        super().__train_loop__(train_loader, train_metrics, test_loader, test_metrics, val_loader,val_metrics, num_epochs,lr_scheduler,
              save_models, model_dir,save_model_interval,display_metrics, save_metrics, notebook_mode, batch_log, save_logs,
              visdom_log,tensorboard_log, save_architecture)


    def __train_func__(self, data):

        self.optimizer.zero_grad()
        if self.clip_grads is not None:

            clip_grads(self.model,self.clip_grads[0],self.clip_grads[1])

        train_x = getattr(data, self.source_field)
        train_y = getattr(data, self.target_field)

        batch_size = get_batch_size(train_x,self.batch_first)

        if isinstance(train_x, list) or isinstance(train_x, tuple):
            train_x = (x.cuda() if self.cuda else x for x in train_x)
        else:
            train_x = train_x.cuda() if self.cuda else train_x

        if isinstance(train_y, list) or isinstance(train_y, tuple):
            train_y = (y.cuda() if self.cuda else y for y in train_y)
        else:
            train_y = train_y.cuda() if self.cuda else train_y

        outputs = self.model(train_x)
        loss = self.loss_fn(outputs, train_y)
        if self.fp16_mode:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        self.optimizer.step()
        self.train_running_loss = self.train_running_loss + (loss.cpu().item() * batch_size)
        
        for metric in self.train_metrics:
            metric.update(outputs, train_y,self.batch_first)

    def __eval_function__(self, data):

        test_x = getattr(data, self.source_field)
        test_y = getattr(data, self.target_field)
        if isinstance(test_x, list) or isinstance(test_x, tuple):
            test_x = (x.cuda() if self.cuda else x for x in test_x)
        else:
            test_x = test_x.cuda() if self.cuda else test_x

        if isinstance(test_y, list) or isinstance(test_y, tuple):
            test_y = (y.cuda() if self.cuda else y for y in test_y)
        else:
            test_y = test_y.cuda() if self.cuda else test_y

        outputs = self.model(test_x)

        for metric in self.test_metrics:
            metric.update(outputs, test_y,self.batch_first)

    def __val_function__(self, data):

        val_x = getattr(data, self.source_field)
        val_y = getattr(data, self.target_field)
        if isinstance(val_x, list) or isinstance(val_x, tuple):
            val_x = (x.cuda() if self.cuda else x for x in val_x)
        else:
            val_x = val_x.cuda() if self.cuda else val_x

        if isinstance(val_y, list) or isinstance(val_y, tuple):
            val_y = (y.cuda() if self.cuda else y for y in val_y)
        else:
            val_y = val_y.cuda() if self.cuda else val_y


        outputs = self.model(val_x)

        for metric in self.val_metrics:
            metric.update(outputs.cpu().data , val_y.cpu().data,self.batch_first)

    def __predict_func__(self, inputs):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            inputs = (x.cuda() if self.cuda else x for x in inputs)
        else:
            inputs = inputs.cuda() if self.cuda else inputs

        return self.model(inputs)

"""returns a complete summary of the model

                   Args:
                       input_sizes: a single tuple or a list of tuples in the case of multiple inputs, specifying the
                       size of the inputs to the model
                       input_types: a single tensor type or a list of tensors in the case of multiple inputs, specifying the 
                       type of the inputs to the model
                       item_length(int): the length of each item in the summary
                       tensorboard_log(str): if enabled, the model will be serialized into a format readable by tensorboard,
                       useful for visualizing the model in tensorboard.
                       """

def summary(self, input_sizes, input_types=torch.FloatTensor, item_length=26, tensorboard_log=None):
    if isinstance(input_sizes, list):
        inputs = (torch.randn(input_size).type(input_type).unsqueeze(0) for input_size, input_type in zip(input_sizes, input_types))

        inputs = (input.cuda() if self.cuda else input for input in inputs)
    else:
        inputs = torch.randn(input_sizes).type(input_types).unsqueeze(0)

        inputs = inputs.cuda() if self.cuda else inputs

    return get_model_summary(self.model, inputs, item_length=item_length, tensorboard_log=tensorboard_log)

"""saves the model in onnx format

                       Args:
                           input_sizes: a single tuple or a list of tuples in the case of multiple inputs, specifying the
                           size of the inputs to the model
                           input_types: a single tensor type or a list of tensors in the case of multiple inputs, specifying the 
                           type of the inputs to the model
                         """
def to_onnx(self, input_sizes, path, input_types=torch.FloatTensor, **kwargs):
    if isinstance(input_sizes, list):
        inputs = (torch.randn(input_size).type(input_type).unsqueeze(0) for input_size, input_type in
                  zip(input_sizes, input_types))

        inputs = (input.cuda() if self.cuda else input for input in inputs)
    else:
        inputs = torch.randn(input_sizes).type(input_types).unsqueeze(0)

        inputs = inputs.cuda() if self.cuda else inputs

    return onnx._export(self.model, inputs, f=path, **kwargs)
