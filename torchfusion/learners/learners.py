import torch
from torch.autograd import Variable
import torch.cuda as cuda
from torch.utils.data import DataLoader
import os
from time import time
from math import ceil
from io import open
from ..utils import PlotInput, visualize, get_model_summary
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.onnx as onnx

r"""Abstract base Model for training, evaluating and performing inference
All custom models should subclass this and implement train, evaluate and predict functions

    Args:
        use_cuda_if_available (boolean): If set to true, training would be done on a gpu if any is available
    
    """


class AbstractBaseLearner():
    def __init__(self, use_cuda_if_available=True):
        self.cuda = False
        if use_cuda_if_available and cuda.is_available():
            self.cuda = True

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
        self.loss_history = []
        self.train_running_loss = None
        self.train_metrics = None
        self.test_metrics = None
        self.val_metrics = None
        self.epochs = 0
        self.iterations = 0
        self.model_dir = os.getcwd()

    r"""Initialize model weights using pre-trained weights from the filepath

        Args:
            path (str): path to a compatible pre-defined model

        """
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)

    r"""Saves the model to the path specified
            Args:
                path (str): path to save model
                save_architecture (boolean): if True, both weights and architecture will be saved, default is False

            """
    def save_model(self, path,save_architecture=False):
        if save_architecture:
            torch.save(self.model,path)
        else:
            torch.save(self.model.state_dict(), path)


    """Complete Training loop
    
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
                model_dir (str) = a path in which to save the models
                notebook_mode (boolean): Optimizes the progress bar for either jupyter notebooks or consoles
                batch_log (boolean): Enables printing of logs at every batch iteration
                save_logs (str): Specifies a filepath in which to permanently save logs at every epoch
                display_metrics (boolean): Enables display of metrics and loss visualizations at the end of each epoch.
                save_metrics (boolean): Enables saving of metrics and loss visualizations at the end of each epoch.
                visdom_log (VisdomLogger): Logs outputs and metrics to the visdom server
                tensorboard_log (str): Logs outputs and metrics to the filepath for visualization in tensorboard
                save_architecture (boolean): Saves the architecture as well as weights during model saving
               
                """
    def train(self,*args):

        self.__train_loop__(*args)

    def __train_loop__(self, train_loader, train_metrics, test_loader=None, test_metrics=None, val_loader=None,val_metrics=None, num_epochs=10,lr_scheduler=None,
              save_models="all", model_dir=os.getcwd(),save_model_interval=1,display_metrics=True, save_metrics=True, notebook_mode=False, batch_log=True, save_logs=None,
              visdom_log=None,tensorboard_log=None, save_architecture=False):

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

                if isinstance(data, list) or isinstance(data, tuple):
                    inputs = data[0]
                else:
                    inputs = data
                batch_size = inputs.size(0)

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
                batch_info = {"train_metrics":self.train_metrics,"train_loss":train_loss}
                for func in self.batch_end_funcs:
                    func(e + 1,i + 1,batch_info)
            if self.cuda:
                cuda.synchronize()
            duration = time() - init_time
            self.epochs += 1

            self.loss_history.append(train_loss)


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
                    print("Val {} : {}".format(metric.name, metric.getValue()))
                    if logfile is not None:
                        logfile.write(os.linesep + "Val {} : {}".format(metric.name, metric.getValue()))


            for metric in self.train_metrics:
                print("Train {} : {}".format(metric.name, metric.getValue()))
                if logfile is not None:
                    logfile.write(os.linesep + "Train {} : {}".format(metric.name, metric.getValue()))

            if logfile is not None:
                logfile.close()

            for metric in self.train_metrics:
                metric.add_history()
            epoch_arr = [x for x in range(e + 1)]

            epoch_arr_tensor = torch.LongTensor(epoch_arr)

            if visdom_log is not None:
                visdom_log.plot_line(torch.FloatTensor(self.loss_history),epoch_arr_tensor,win="train_loss",title="Train Loss")

                if test_metrics is not None:
                     for metric in test_metrics:
                         visdom_log.plot_line(torch.FloatTensor(metric.history),epoch_arr_tensor,win="test_{}".format(metric.name),title="Test {}".format(metric.name))
                if val_metrics is not None:
                     for metric in val_metrics:
                         visdom_log.plot_line(torch.FloatTensor(metric.history),epoch_arr_tensor,win="val_{}".format(metric.name),title="Val {}".format(metric.name))

                for metric in train_metrics:
                    visdom_log.plot_line(torch.FloatTensor(metric.history), epoch_arr_tensor,
                                         win="train_{}".format(metric.name), title="Train {}".format(metric.name))



            if tensorboard_log is not None:
                writer = SummaryWriter(os.path.join(model_dir,tensorboard_log))
                for epoch in epoch_arr:
                    writer.add_scalar("logs/train_loss",self.loss_history[epoch],global_step=epoch+1)

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
                visualize(epoch_arr, [PlotInput(value=self.loss_history, name="Train Loss", color="red")],
                          display=display_metrics,
                          save_path=save_path)

            if test_loader is not None and (display_metrics or save_metrics):
                for metric in self.test_metrics:

                    save_path = None

                    if save_metrics:
                        save_path = os.path.join(model_dir, "test_{}_epoch_{}.png".format(metric.name, e + 1))
                    visualize(epoch_arr, [PlotInput(value=metric.history, name="Test " + metric.name, color="blue")],
                              display=display_metrics,
                              save_path=save_path)

            if val_loader is not None and (display_metrics or save_metrics):
                for metric in self.val_metrics:

                    save_path = None

                    if save_metrics:
                        save_path = os.path.join(model_dir, "val_{}_epoch_{}.png".format(metric.name, e + 1))
                    visualize(epoch_arr, [PlotInput(value=metric.history, name="Val " + metric.name, color="blue")],
                              display=display_metrics,
                              save_path=save_path)

            for metric in self.train_metrics:
                save_path = None
                if save_metrics:
                    save_path = os.path.join(model_dir, "train_{}_epoch_{}.png".format(metric.name, e + 1))
                visualize(epoch_arr, [PlotInput(value=metric.history, name="Train " + metric.name, color="blue")],
                              display=display_metrics,
                              save_path=save_path)
            batch_info = {"train_metrics": self.train_metrics, "train_loss": train_loss,"duration":duration}
            if self.test_metrics != None and test_loader != None:
                batch_info["test_metrics"] = self.test_metrics

            if self.val_metrics != None and val_loader != None:
                batch_info["val_metrics"] = self.val_metrics
            for func in self.epoch_end_funcs:
                func(e + 1,batch_info)

        train_end_time = time() - train_start_time
        batch_info = {"train_metrics": self.train_metrics,"train_duration":train_end_time}
        if self.test_metrics != None and test_loader != None:
            batch_info["test_metrics"] = self.test_metrics

        if val_loader != None:
            batch_info["val_metrics"] = self.val_metrics

        for func in self.train_completed_funcs:
            func(batch_info)



    r"""Training logic, all models must override this
            Args:
                data: a single batch of data from the train_loader
    """

    def __train_func__(self, data):
        raise NotImplementedError()

    """ Evaluation function to evaluate performance on a test set.

            Arguments:
                test_loader: An instance of the DataLoader class
                metrics: The metrics to evaluate the test_loader
                optimizer: the optimizer
                """
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
        for metric in self.test_metrics:
            metric.add_history()


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
        for metric in self.val_metrics:
            metric.add_history()


    r"""Validation logic, all models must override this
            Args:
                data: a single batch of data from the test_loader
        """

    def __val_function__(self, data):
        raise NotImplementedError()

    r"""Training logic, all models must override this
            Args:
                data: a single batch of data from the train_loader
    """
    def predict(self, inputs, apply_softmax=False):
        self.model.eval()

        if isinstance(inputs, DataLoader):
            predictions = []
            for i, data in enumerate(inputs):
                batch_pred = self.__predict_func__(data)
                if apply_softmax:
                    batch_pred = torch.nn.Softmax(dim=1)(batch_pred)

                for pred in batch_pred:
                    predictions.append(pred)

            return predictions
        else:
            pred = self.__predict_func__(inputs)
            if apply_softmax:
                pred = torch.nn.Softmax(dim=1)(pred)
            return pred

    r"""Inference logic, all models must override this
            Args:
                data: a batch of data 
    """

    def __predict_func__(self, input):
        raise NotImplementedError()


class BaseTextLearner(BaseLearner):
    def __init__(self, model, source_field, target_field, batch_first=False,use_cuda_if_available=True):

        super(BaseTextLearner, self).__init__(model,use_cuda_if_available)

        self.batch_first = batch_first
        self.source_field = source_field
        self.target_field = target_field



    def __train_loop__(self, train_loader, train_metrics, test_loader=None, test_metrics=None, val_loader=None,val_metrics=None, num_epochs=10,lr_scheduler=None,
              save_models="all", model_dir=os.getcwd(),save_model_interval=1,display_metrics=True, save_metrics=True, notebook_mode=False, batch_log=True, save_logs=None,
              visdom_log=None,tensorboard_log=None, save_architecture=False):

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

                if self.batch_first:
                    batch_size = source.size(0)

                else:
                    batch_size = source.size(len(source.size()) - 1)

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
                batch_info = {"train_metrics":self.train_metrics,"train_loss":train_loss}
                for func in self.batch_end_funcs:
                    func(e + 1,i + 1,batch_info)
            if self.cuda:
                cuda.synchronize()
            duration = time() - init_time
            self.epochs += 1

            self.loss_history.append(train_loss)


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
                    print("Val {} : {}".format(metric.name, metric.getValue()))
                    if logfile is not None:
                        logfile.write(os.linesep + "Val {} : {}".format(metric.name, metric.getValue()))


            for metric in self.train_metrics:
                print("Train {} : {}".format(metric.name, metric.getValue()))
                if logfile is not None:
                    logfile.write(os.linesep + "Train {} : {}".format(metric.name, metric.getValue()))

            if logfile is not None:
                logfile.close()

            for metric in self.train_metrics:
                metric.add_history()
            epoch_arr = [x for x in range(e + 1)]

            epoch_arr_tensor = torch.LongTensor(epoch_arr)

            if visdom_log is not None:
                visdom_log.plot_line(torch.FloatTensor(self.loss_history),epoch_arr_tensor,win="train_loss",title="Train Loss")

                if test_metrics is not None:
                     for metric in test_metrics:
                         visdom_log.plot_line(torch.FloatTensor(metric.history),epoch_arr_tensor,win="test_{}".format(metric.name),title="Test {}".format(metric.name))
                if val_metrics is not None:
                     for metric in val_metrics:
                         visdom_log.plot_line(torch.FloatTensor(metric.history),epoch_arr_tensor,win="val_{}".format(metric.name),title="Val {}".format(metric.name))

                for metric in train_metrics:
                    visdom_log.plot_line(torch.FloatTensor(metric.history), epoch_arr_tensor,
                                         win="train_{}".format(metric.name), title="Train {}".format(metric.name))



            if tensorboard_log is not None:
                writer = SummaryWriter(os.path.join(model_dir,tensorboard_log))
                for epoch in epoch_arr:
                    writer.add_scalar("logs/train_loss",self.loss_history[epoch],global_step=epoch+1)

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
                visualize(epoch_arr, [PlotInput(value=self.loss_history, name="Train Loss", color="red")],
                          display=display_metrics,
                          save_path=save_path)

            if test_loader is not None and (display_metrics or save_metrics):
                for metric in self.test_metrics:

                    save_path = None

                    if save_metrics:
                        save_path = os.path.join(model_dir, "test_{}_epoch_{}.png".format(metric.name, e + 1))
                    visualize(epoch_arr, [PlotInput(value=metric.history, name="Test " + metric.name, color="blue")],
                              display=display_metrics,
                              save_path=save_path)

            if val_loader is not None and (display_metrics or save_metrics):
                for metric in self.val_metrics:

                    save_path = None

                    if save_metrics:
                        save_path = os.path.join(model_dir, "val_{}_epoch_{}.png".format(metric.name, e + 1))
                    visualize(epoch_arr, [PlotInput(value=metric.history, name="Val " + metric.name, color="blue")],
                              display=display_metrics,
                              save_path=save_path)

            for metric in self.train_metrics:
                save_path = None
                if save_metrics:
                    save_path = os.path.join(model_dir, "train_{}_epoch_{}.png".format(metric.name, e + 1))
                visualize(epoch_arr, [PlotInput(value=metric.history, name="Train " + metric.name, color="blue")],
                              display=display_metrics,
                              save_path=save_path)
            batch_info = {"train_metrics": self.train_metrics, "train_loss": train_loss,"duration":duration}
            if self.test_metrics != None and test_loader != None:
                batch_info["test_metrics"] = self.test_metrics

            if self.val_metrics != None and val_loader != None:
                batch_info["val_metrics"] = self.val_metrics
            for func in self.epoch_end_funcs:
                func(e + 1,batch_info)

        train_end_time = time() - train_start_time
        batch_info = {"train_metrics": self.train_metrics,"train_duration":train_end_time}
        if self.test_metrics != None and test_loader != None:
            batch_info["test_metrics"] = self.test_metrics

        if val_loader != None:
            batch_info["val_metrics"] = self.val_metrics

        for func in self.train_completed_funcs:
            func(batch_info)



class BaseLearnerCore(BaseLearner,BaseTextLearner):

    def summary(self,input_size,input_type=torch.FloatTensor,item_length=26,tensorboard_log=None):
        input = torch.randn(input_size).type(input_type).unsqueeze(0)

        if self.cuda:
            input.cuda()

        return get_model_summary(self.model,Variable(input),item_length=item_length,tensorboard_log=tensorboard_log)

    def to_onnx(self,input_size,path,input_type=torch.FloatTensor,**kwargs):
        input = Variable(torch.randn(input_size).type(input_type).unsqueeze(0))

        if self.cuda:
            input.cuda()

        return onnx._export(self.model, input, f=path, **kwargs)



class StandardLearner(BaseLearnerCore):
    def __init__(self, model, use_cuda_if_available=True):
        super().__init__(model, use_cuda_if_available)

    def train(self, train_loader, loss_fn, optimizer, train_metrics, test_loader=None, test_metrics=None, val_loader=None,val_metrics=None, num_epochs=10,lr_scheduler=None,
              save_models="all", model_dir=os.getcwd(),save_model_interval=1,display_metrics=False, save_metrics=False, notebook_mode=False, batch_log=True, save_logs=None,
              visdom_log=None,tensorboard_log=None, save_architecture=False):

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        super().__train_loop__(train_loader, train_metrics, test_loader, test_metrics, val_loader,val_metrics, num_epochs,lr_scheduler,
              save_models, model_dir,save_model_interval,display_metrics, save_metrics, notebook_mode, batch_log, save_logs,
              visdom_log,tensorboard_log, save_architecture)

    def __train_func__(self, data):

        self.optimizer.zero_grad()

        train_x, train_y = data

        batch_size = train_x.size(0)
        if self.cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
        train_x = Variable(train_x)
        train_y = Variable(train_y)
        outputs = self.model(train_x)
        loss = self.loss_fn(outputs, train_y)
        loss.backward()

        self.optimizer.step()

        self.train_running_loss.add_(loss.cpu() * batch_size)

        for metric in self.train_metrics:
            metric.update(outputs.cpu().data, train_y.cpu().data)

    def __eval_function__(self, data):

        test_x, test_y = data
        if self.cuda:
            test_x = test_x.cuda()
            test_y = test_y.cuda()
        test_x = Variable(test_x)
        test_y = Variable(test_y)

        outputs = self.model(test_x)

        for metric in self.test_metrics:
            metric.update(outputs.cpu().data, test_y.cpu().data)

    def __val_function__(self, data):

        val_x, val_y = data
        if self.cuda:
            val_x = val_x.cuda()
            val_y = val_y.cuda()
        val_x = Variable(val_x)
        val_y = Variable(val_y)

        outputs = self.model(val_x)

        for metric in self.val_metrics:
            metric.update(outputs.cpu().data, val_y.cpu().data)

    def __predict_func__(self, inputs):

        return self.model(Variable(inputs.cuda() if self.cuda else inputs))


#TO-DO CONFIRM LABEL DIMENSIONS ISSUE, CHECK IF EXPANSION IS NEEDED
#modify predict function to account for torchtext in the text classifier
#RUN TESTS TO VERIFY PREDICTION BEHAVIOUR


class TextClassifier(BaseLearnerCore):
    def __init__(self, model, source_field, target_field, batch_first=False, use_cuda_if_available=True):
        super(TextClassifier, self).__init__(model, source_field, target_field, batch_first, use_cuda_if_available)

    def train(self, train_loader, loss_fn, optimizer, train_metrics, test_loader=None, test_metrics=None, val_loader=None,val_metrics=None, num_epochs=10,lr_scheduler=None,
              save_models="all", model_dir=os.getcwd(),save_model_interval=1,display_metrics=False, save_metrics=False, notebook_mode=False, batch_log=True, save_logs=None,
              visdom_log=None,tensorboard_log=None, save_architecture=False):

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        super().__train_loop__(train_loader, train_metrics, test_loader, test_metrics, val_loader,val_metrics, num_epochs,lr_scheduler,
              save_models, model_dir,save_model_interval,display_metrics, save_metrics, notebook_mode, batch_log, save_logs,
              visdom_log,tensorboard_log, save_architecture)


    def __train_func__(self, data):

        self.optimizer.zero_grad()

        train_x = getattr(data, self.source_field)
        train_y = getattr(data, self.target_field)

        if self.batch_first:
            batch_size = train_x.size(0)
        else:
            batch_size = train_x.size(len(train_x.size()) - 1)

        if self.cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
        train_x = Variable(train_x)
        train_y = Variable(train_y)
        outputs = self.model(train_x)
        loss = self.loss_fn(outputs, train_y)
        loss.backward()

        self.optimizer.step()

        self.train_running_loss.add_(loss.cpu() * batch_size)

        for metric in self.train_metrics:
            metric.update(outputs.cpu().data, train_y.cpu().data)

    def __eval_function__(self, data):

        test_x = getattr(data, self.source_field)
        test_y = getattr(data, self.target_field)
        if self.cuda:
            test_x = test_x.cuda()
            test_y = test_y.cuda()
        test_x = Variable(test_x)
        test_y = Variable(test_y)

        outputs = self.model(test_x)

        for metric in self.test_metrics:
            metric.update(outputs.cpu().data, test_y.cpu().data)

    def __val_function__(self, data):

        val_x = getattr(data, self.source_field)
        val_y = getattr(data, self.target_field)
        if self.cuda:
            val_x = val_x.cuda()
            val_y = val_y.cuda()
        val_x = Variable(val_x)
        val_y = Variable(val_y)

        outputs = self.model(val_x)

        for metric in self.val_metrics:
            metric.update(outputs.cpu().data, val_y.cpu().data)

    def __predict_func__(self, inputs):

        return self.model(Variable(inputs.cuda() if self.cuda else inputs))
