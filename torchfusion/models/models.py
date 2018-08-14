import torch
from torch.autograd import Variable
import torch.cuda as cuda
from torch.utils.data import DataLoader
import os
from time import time
from math import ceil
from io import open
from ..utils import PlotInput, visualize, get_model_summary,adjust_learning_rate
from tensorboardX import SummaryWriter
import torch.onnx as onnx
from deprecation import deprecated
from .. import __version__


r"""This is the base learner for training, evaluating and performing inference with a single model
All custom learners should subclass this and implement __train, __evaluate and __predict functions
This class already takes care of data loading, iterations and metrics, subclasses only need to define custom logic for training,
evaluation and prediction

    Args:
        model (nn.Module):  the module to be used for training, evaluation and inference.
        use_cuda_if_available (boolean): If set to true, training would be done on a gpu if any is available
"""


class BaseModel(object):
    def __init__(self, model, use_cuda_if_available=True):
        self.model = model
        self.cuda = False
        if use_cuda_if_available and cuda.is_available():
            self.cuda = True

        self.loss_history = []
        self.__input_hooks = []
        self.visdom_log = None
        self.tensorbord_log = None


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

    def save_model(self, path, save_architecture=False):
        if save_architecture:
            torch.save(self.model, path)
        else:
            torch.save(self.model.state_dict(), path)

    def to_onnx(self, inputs, path, **kwargs):

        if isinstance(inputs, tuple):
            in_tensors = []
            for input in inputs:
                in_tensors.append(Variable(input.cuda() if self.cuda else input))
        else:
            in_tensors = Variable(inputs.cuda() if self.cuda else inputs)

        return onnx._export(self.model, in_tensors, f=path, **kwargs)

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

    @deprecated(deprecated_in="0.2.0", removed_in="0.4.0", current_version=__version__,
                details="Use the StandardLearner instead")
    def train(self, train_loader, loss_fn, optimizer, train_metrics, test_loader=None, test_metrics=None,
              num_epochs=10, lr_schedule=None,
              save_models="all", model_dir=os.getcwd(),notebook_mode=False,
              batch_log=True, save_logs=None, display_metrics=True, save_metrics=True,
              visdom_log=None, tensorboard_log=None, save_architecture=False):

        if save_models not in ["all", "best"]:
            raise ValueError("save models must be 'all' or 'best' , {} is invalid".format(save_models))
        if save_models == "best" and test_loader is None and train_loader is None:
            raise ValueError("save models can only be best when test_loader or val_loader is provided ")

        if test_loader is not None:
            if test_metrics is None:
                raise ValueError("You must provide a metric for your test data")
            elif len(test_metrics) == 0:
                raise ValueError("test metrics cannot be an empty list")

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        models_all = os.path.join(model_dir, "all_models")
        models_best = os.path.join(model_dir, "best_models")

        if not os.path.exists(models_all):
            os.mkdir(models_all)

        if not os.path.exists(models_best) and test_loader is not None:
            os.mkdir(models_best)

        self.tensorbord_log = tensorboard_log
        self.visdom_log = visdom_log

        from tqdm import tqdm_notebook
        from tqdm import tqdm

        best_test_metric = 0.0
        train_start_time = time()
        for e in range(num_epochs):

            print("Epoch {} of {}".format(e + 1, num_epochs))

            for metric in train_metrics:
                metric.reset()

            self.model.train()

            self.on_epoch_start(e+1)

            running_loss = torch.Tensor([0.0])
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
                self.on_batch_start(e+1,i+1)

                if isinstance(data, list) or isinstance(data, tuple):
                    inputs = data[0]
                else:
                    inputs = data
                batch_size = inputs.size(0)

                if max_batch_size < batch_size:
                    max_batch_size = batch_size


                if len(self.__input_hooks) > 0:

                    for hook in self.__input_hooks:
                        inputs = hook(inputs)

                if isinstance(data, list):
                    data[0] = inputs
                elif isinstance(data, tuple):
                    data = (inputs,data[1])
                else:
                    data = inputs


                self.__train_func__(data,optimizer,loss_fn,train_metrics,running_loss,e+1,i+1)
                data_len += batch_size
                train_loss = running_loss.item() / data_len

                if batch_log:
                    progress_message = ""
                    for metric in train_metrics:
                        progress_message += "Train {} : {}".format(metric.name, metric.getValue())
                    progress_.set_description("{}/{} batches ".format(int(ceil(data_len / max_batch_size)),
                                                                      int(ceil(len(
                                                                          train_loader.dataset) / max_batch_size))))
                    progress_dict = {"Train Loss": train_loss}
                    for metric in train_metrics:
                        progress_dict["Train " + metric.name] = metric.getValue()

                    progress_.set_postfix(progress_dict)
                self.on_batch_end(e,i+1,train_metrics,train_loss)
            if self.cuda:
                cuda.synchronize()
            duration = time() - init_time

            if lr_schedule is not None:
                lr = lr_schedule(e)
                adjust_learning_rate(lr,optimizer)



            self.loss_history.append(train_loss)

            model_file = os.path.join(models_all, "model_{}.pth".format(e + 1))

            if save_models == "all":
                self.save_model(model_file, save_architecture)

            logfile = None
            if save_logs is not None:
                logfile = open(save_logs, "a")

            print(os.linesep + "Epoch: {}, Duration: {} , Train Loss: {}".format(e + 1, duration, train_loss))
            if logfile is not None:
                logfile.write(
                    os.linesep + "Epoch: {}, Duration: {} , Train Loss: {}".format(e + 1, duration, train_loss))



            if test_loader is not None:
                message = "Test Accuracy did not improve, current best is {}".format(best_test_metric)
                current_best = best_test_metric
                self.evaluate(test_loader, test_metrics)
                result = self.test_metrics[0].getValue()

                if result > current_best:
                    best_test_metric = result
                    message = "Test {} improved from {} to {}".format(test_metrics[0].name, current_best, result)
                    model_file = os.path.join(models_best, "model_{}.pth".format(e + 1))
                    self.save_model(model_file, save_architecture)

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


            for metric in train_metrics:
                print("Train {} : {}".format(metric.name, metric.getValue()))
                if logfile is not None:
                    logfile.write(os.linesep + "Train {} : {}".format(metric.name, metric.getValue()))

            if logfile is not None:
                logfile.close()

            for metric in train_metrics:
                metric.add_history()

            self.on_epoch_end(e, train_metrics, test_metrics, train_loss, duration)

            epoch_arr = [x for x in range(e + 1)]
            epoch_arr_tensor = torch.LongTensor(epoch_arr)

            if visdom_log is not None:
                visdom_log.plot_line(torch.FloatTensor(self.loss_history), epoch_arr_tensor, win="train_loss",
                                     title="Train Loss")

                if test_metrics is not None:
                    for metric in test_metrics:
                        visdom_log.plot_line(torch.FloatTensor(metric.history), epoch_arr_tensor,
                                             win="test_{}".format(metric.name), title="Test {}".format(metric.name))

                for metric in train_metrics:
                    visdom_log.plot_line(torch.FloatTensor(metric.history), epoch_arr_tensor,
                                         win="train_{}".format(metric.name), title="Train {}".format(metric.name))

            if tensorboard_log is not None:
                writer = SummaryWriter(os.path.join(model_dir, tensorboard_log))
                for epoch in epoch_arr:
                    writer.add_scalar("logs/train_loss", self.loss_history[epoch], global_step=epoch)

                if test_metrics is not None:
                    for metric in test_metrics:
                        writer.add_scalar("logs/test_metrics/{}".format(metric.name), metric.getValue(),
                                          global_step=e + 1)

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


            for metric in train_metrics:
                save_path = None
                if save_metrics:
                    save_path = os.path.join(model_dir, "train_{}_epoch_{}.png".format(metric.name, e + 1))
                visualize(epoch_arr, [PlotInput(value=metric.history, name="Train " + metric.name, color="blue")],
                              display=display_metrics,
                              save_path=save_path)

        train_end_time = time() - train_start_time

        self.on_training_completed(train_metrics, test_metrics, train_end_time)

    r"""Training logic, all models must override this
            Args:
                data: a single batch of data from the train_loader
    """

    def __train_func__(self, data,optimizer,loss_fn,train_metrics,running_loss,epoch,batch_num):
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

        for metric in metrics:
            metric.reset()

        self.model.eval()

        for i, data in enumerate(test_loader):
            self.__eval_function__(data,metrics)
        for metric in metrics:
            metric.add_history()

    r"""Evaluation logic, all models must override this
            Args:
                data: a single batch of data from the test_loader
        """

    def __eval_function__(self, data,metrics):
        raise NotImplementedError()

    r"""Validates the dataset on the set of provided metrics
                Args:
                    val_loader (DataLoader): an instance of DataLoader containing the test set
                    metrics ([]): an array of metrics for evaluating the test set
            """

    def predict(self, inputs, apply_softmax=False):
        self.model.eval()

        if isinstance(inputs, DataLoader):
            predictions = []
            for i, data in enumerate(inputs):
                pred = self.__predict_func__(data)
                if apply_softmax:
                    pred = torch.nn.Softmax(dim=1)(pred)
                predictions.append(pred)

            output_array = []

            for batch in predictions:
                for pred in batch:
                    output_array.append(pred)
            return output_array
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

    """ Adds an hook that transforms the inputs before being passed into __train_func__

        The function should take in the input and return the transformed version
        Example:

            def transform_hook(input):

                return input * 0.5

        """

    def add_input_hook(self, function):
        self.__input_hooks.append(function)

    """ Callback that is invoked at the start of each epoch,
        epoch: the current epoch
        """

    def on_epoch_start(self, epoch):
        pass

    """ Callback that is invoked at the end of each epoch,
           epoch: the current epoch
           train_metrics: The train metrics
           train_loss: the current loss value
           duration: Total duration of the current epoch
           """

    def on_epoch_end(self, epoch, train_metrics, test_metrics, train_loss, duration):
        pass

    """ Callback that is invoked at the start of each batch,
            epoch: the current epoch
            batch_num: The current batch index
    """

    def on_batch_start(self, epoch, batch_num):
        pass

    """ Callback that is invoked at the end of each batch,
              epoch: the current epoch
              batch_num: the current batch index
              train_metrics: The train metrics
              train_loss: the current running loss value
               """

    def on_batch_end(self, epoch, batch_num, train_metrics, train_loss):
        pass

    """ Callback that is invoked at the end of training,
              train_metrics: the train metrics
              test_metrics: the test metrics
              train_duration: the total duration of training
               """

    def on_training_completed(self, train_metrics, test_metrics, train_duration):
        pass

    def summary(self, input_size, input_type=torch.FloatTensor, item_length=26, tensorboard_log=None):
        input = torch.randn(input_size).type(input_type).unsqueeze(0)

        if self.cuda:
            input.cuda()

        return get_model_summary(self.model, Variable(input), item_length=item_length, tensorboard_log=tensorboard_log)


class StandardModel(BaseModel):

    def __init__(self, model, use_cuda_if_available=True):
        super(StandardModel, self).__init__(model, use_cuda_if_available)


    def __train_func__(self, data,optimizer,loss_fn,train_metrics,running_loss,epoch,batch_num):

        optimizer.zero_grad()

        train_x, train_y = data

        batch_size = train_x.size(0)
        if self.cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
        train_x = Variable(train_x)
        train_y = Variable(train_y)
        outputs = self.model(train_x)
        loss = loss_fn(outputs, train_y)
        loss.backward()

        optimizer.step()

        running_loss.add_(loss.cpu() * batch_size)

        for metric in train_metrics:
            metric.update(outputs.cpu().data, train_y.cpu().data)

    def __eval_function__(self, data,metrics):

        test_x, test_y = data
        if self.cuda:
            test_x = test_x.cuda()
            test_y = test_y.cuda()
        test_x = Variable(test_x)
        test_y = Variable(test_y)

        outputs = self.model(test_x)

        for metric in metrics:
            metric.update(outputs.cpu().data, test_y.cpu().data)

    def __predict_func__(self, inputs):

        if isinstance(inputs, list):
            in_tensors = [Variable(input.cuda() if self.cuda else input) for input in inputs]
        else:
            in_tensors = [Variable(inputs.cuda() if self.cuda else inputs)]

        output = self.model(*in_tensors)

        return output

    def to_onnx(self, input_size, path, input_type=torch.FloatTensor, **kwargs):
        input = torch.randn(input_size).type(input_type).unsqueeze(0)

        if self.cuda:
            input.cuda()

        return super(StandardModel, self).to_onnx(Variable(input), path=path, **kwargs)

