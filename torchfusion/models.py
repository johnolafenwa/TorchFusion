import torch
from torch.autograd import Variable
import torch.cuda as cuda
from torch.utils.data import DataLoader
from .utils import *
import os
from time import time
from collections import namedtuple
import torch.nn as nn
from math import ceil
from io import open
from .utils import PlotInput,visualize

"""This is the base Model for training, evaluating and performing inference
All custom trainers that use a single model should subclass this 

Arguments:
    model:  the module to be used for training, evaluation and inference.
    use_cuda_if_available: If set to true, training would be done on a gpu if any is available"""

class BaseModel():
    def __init__(self, model,use_cuda_if_available=True):
        self.model = model
        self.cuda = False
        if use_cuda_if_available and cuda.is_available():
            self.cuda = True
        self.loss_history = []
        self.__input_hooks = []

    """Loads a pretrained model

    Arguments:
        path: a filepath or read-stream of an existing pre-trained model """

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)

    """ Saves the model

        Arguments:
            path: a filepath or write-stream to save the trained model"""

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    """trainer function for the model

            Arguments:
                train_loader: an instance of DataLoader containing the training set
                loss_fn: the loss function 
                optimizer: an optimizer for updating parameters 
                train_metrics: an array of metrics for evaluating the training set
                test_loader: an instance of DataLoader containing the test set
                test_metrics: an array of metrics for evaluating the test set
                num_epochs: The maximum number of training epochs
                lr_schedule: An lr scheduler function that specifies the learning rate for each epoch
                   '''EXAMPLE'''
                   def lr_function(epoch):
                        lr = 0.1
                        if epoch > 90:
                            lr /= 1000
                        elif epoch > 60:
                            lr /= 100
                        elif epoch > 30:
                            lr /= 10
                        
                        return lr
                    
                Note: The function is called at the end of each epoch
                save_models: If all, the model is saved at the end of each epoch while the best models based 
                    on the test set are also saved in best_models folder
                    If 'best', only the best models are saved,  test_loader and test_metrics must be provided
                model_dir = a path in which to save the models
                notebook_mode: Optimizes the progress bar for either jupyter notebooks or consoles
                batch_log: Enables printing of logs at every batch iteration
                save_logs: Specifies a filepath in which to permanently save logs at every epoch
                display_metrics: Enables display of metrics and loss visualizations at the end of each epoch.
                save_metrics: Enables saving of metrics and loss visualizations at the end of each epoch.
            """
    def train(self, train_loader, loss_fn, optimizer,train_metrics,test_loader=None,test_metrics=None, num_epochs=10, lr_schedule=None,
              save_models="all", model_dir=os.getcwd(),notebook_mode=False,batch_log=True,save_logs=None,display_metrics=True,save_metrics=True):


        if save_models not in ["all", "best"]:
            raise ValueError("save models must be 'all' or 'best' , {} is invalid".format(save_models))
        if save_models == "best" and test_loader is None:
            raise ValueError("save models can only be best when testloader is provided")

        if test_loader is not None:
            if test_metrics is None:
                raise ValueError("You must provide a metric for your test data")
            elif len(test_loader) == 0:
                raise ValueError("test metrics cannot be an empty list")

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)


        models_all = os.path.join(model_dir, "all_models")
        models_best = os.path.join(model_dir, "best_models")


        if not os.path.exists(models_all):
            os.mkdir(models_all)

        if not os.path.exists(models_best) and test_loader is not None:
            os.mkdir(models_best)


        from tqdm import tqdm_notebook
        from tqdm import tqdm

        best_metric = 0.0
        train_start_time = time()
        for e in tqdm(range(num_epochs)):
            print("Epoch {} of {}".format(e,num_epochs))

            for metric in train_metrics:
                metric.reset()

            self.model.train()
            self.on_epoch_start(e)

            running_loss = torch.Tensor([0.0])
            train_loss = 0.0
            data_len = 0


            if notebook_mode and batch_log:
                progress_ = tqdm_notebook(enumerate(train_loader))
            elif batch_log:
                progress_ = tqdm(enumerate(train_loader))
            else:
                progress_ = enumerate(train_loader)

            main_batch_size = 0

            init_time = time()

            for i, data in progress_:
                self.on_batch_start(e, i)

                if isinstance(data, list) or isinstance(data, tuple):
                    inputs = data[0]
                else:
                    inputs = data
                batch_size = inputs.size(0)

                if main_batch_size < batch_size:
                    main_batch_size = batch_size
                if len(self.__input_hooks) > 0:

                    for hook in self.__input_hooks:
                        inputs = hook(inputs)

                if isinstance(data, list):
                    data[0] = inputs
                elif isinstance(data, tuple):
                    data = (inputs,data[1])
                else:
                    data = inputs

                self.__train_func__(data,optimizer,loss_fn,train_metrics,running_loss,e,i)

                data_len += batch_size
                train_loss = running_loss.item()/data_len

                if batch_log:
                    progress_message = ""
                    for metric in train_metrics:
                        progress_message += "Train {} : {}".format(metric.name, metric.getValue())
                    progress_.set_description("{}/{} batches ".format(int(ceil(data_len / main_batch_size)),
                                                              int(ceil(len(train_loader.dataset) / main_batch_size))))
                    progress_dict = {"Train Loss": train_loss}
                    for metric in train_metrics:
                        progress_dict["Train " + metric.name] = metric.getValue()

                    progress_.set_postfix(progress_dict)

                self.on_batch_end(e, i, train_metrics, train_loss)
            if self.cuda:
                cuda.synchronize()

            self.loss_history.append(train_loss)
            duration = time() - init_time

            if lr_schedule is not None:
                lr = lr_schedule(e)
                adjust_learning_rate(lr,optimizer)

            model_file = os.path.join(models_all, "model_{}.pth".format(e))
            self.save_model(model_file)

            logfile = None
            if save_logs is not None:
                logfile = open(save_logs,"a")


            print(os.linesep+"Epoch: {}, Duration: {} , Train Loss: {}".format(e, duration, train_loss))
            if logfile is not None:
                logfile.write(os.linesep+"Epoch: {}, Duration: {} , Train Loss: {}".format(e, duration, train_loss))

            if test_loader is not None:
                message = "Accuracy did not improve"
                current_best = best_metric
                self.evaluate(test_loader,test_metrics)
                result = test_metrics[0].getValue()
                if result > current_best:
                    best_metric = result
                    message = "{} improved from {} to {}".format(test_metrics[0].name,current_best, result)
                    model_file = os.path.join(models_best,"model_{}.pth".format(e))
                    self.save_model(model_file)

                    print(os.linesep+"{} New Best Model saved in {}".format(message,model_file))
                    if logfile is not None:
                        logfile.write(os.linesep+"{} New Best Model saved in {}".format(message,model_file))

                else:
                    print(os.linesep+message)
                    if logfile is not None:
                        logfile.write(os.linesep+message)

                for metric in test_metrics:
                    print("Test {} : {}".format(metric.name,metric.getValue()))
                    if logfile is not None:
                        logfile.write(os.linesep+"Test {} : {}".format(metric.name,metric.getValue()))


            for metric in train_metrics:
                print("Train {} : {}".format(metric.name, metric.getValue()))
                if logfile is not None:
                    logfile.write(os.linesep + "Train {} : {}".format(metric.name, metric.getValue()))

            if logfile is not None:
                logfile.close()

            for metric in train_metrics:
                metric.add_history()


            epoch_arr = [x for x in range(e+1)]

            if display_metrics or save_metrics:

                save_path = None

                if save_metrics:
                    save_path = os.path.join(model_dir, "epoch_{}_loss.png".format(e))
                visualize(epoch_arr, [PlotInput(value=self.loss_history, name="Train Loss", color="red")],display=display_metrics,
                          save_path=save_path)

            if test_loader is not None and (display_metrics or save_metrics):
                    for metric in test_metrics:

                        save_path = None

                        if save_metrics:
                            save_path = os.path.join(model_dir, "test_{}_epoch_{}.png".format(metric.name, e))
                        visualize(epoch_arr, [PlotInput(value=metric.history, name="Test "+metric.name, color="blue")],display=display_metrics,
                                      save_path=save_path)
            for metric in train_metrics:
                if save_metrics:
                    save_path = os.path.join(model_dir, "train_{}_epoch_{}.png".format(metric.name, e))
                visualize(epoch_arr, [PlotInput(value=metric.history, name="Train " + metric.name, color="blue")],display=display_metrics,
                          save_path=save_path)

            self.on_epoch_end(e, train_metrics, test_metrics, train_loss, duration)
        train_end_time = time() - train_start_time

        self.on_training_completed(train_metrics,test_metrics,train_end_time)

    """ Abstract function containing the training logic, all custom trainers must override this.

            Arguments:
                data: a single batch of data from the training set
                loss_fn: the loss function
                optimizer: the optimizer
                train_metrics: array of metrics for training
                running_loss: the current running loss
                epoch: current epoch
                batch_num: the current batch index
                """
    def __train_func__(self,data,optimizer,loss_fn,train_metrics,running_loss,epoch,batch_num):
        raise NotImplementedError()

    """ Evaluation function to evaluate performance on a test set.

            Arguments:
                test_loader: An instance of the DataLoader class
                metrics: The metrics to evaluate the test_loader
                optimizer: the optimizer
                """
    def evaluate(self, test_loader, metrics):
        for metric in metrics:
            metric.reset()

        self.model.eval()

        for i, data in enumerate(test_loader):
            self.__eval_function__(data,metrics)
        for metric in metrics:
            metric.add_history()

    """ Abstract function containing the custom logic for evaluating the test set,all custom trainers must override this.

            Arguments:
                data: a single batch from the test set
                metrics: The metrics to evaluate the data
                """
    def __eval_function__(self,data,metrics):
        raise NotImplementedError()

    """ Returns predictions for a given input tensor or a an instance of a DataLoader

            Arguments:
               inputs: input Tensor or DataLoader
               apply_softmax: If True, applies softmax to the output
                """
    def predict(self, inputs,apply_softmax=False):
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

    """ Abstract function containing custom logic for performing inference, must return the output,
    all custom trainers should implement this.
    input: An input tensor or a batch of input tensors
    """

    def __predict_func__(self,input):
        raise NotImplementedError()

    """ Returns a detailed summary of the model, including input and output sizes, computational cost in flops,
    number of parameters and number of layers of each type
        input_size: The size of the input tensor
        """
    def summary(self,input_size):

        input = torch.randn(input_size)

        input = input.unsqueeze(0)
        if self.cuda:
            input = input.cuda()
        input = Variable(input)

        summary = []
        ModuleDetails = namedtuple("Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
        hooks = []
        layer_instances = {}
        def add_hooks(module):

            def hook(module, input, output):

                class_name = str(module.__class__.__name__)

                instance_index = 1
                if class_name not in layer_instances:
                    layer_instances[class_name] = instance_index
                else:
                    instance_index = layer_instances[class_name] + 1
                    layer_instances[class_name] = instance_index

                layer_name = class_name + "_" + str(instance_index)

                params = 0
                if hasattr(module, "weight"):
                    weight_size = module.weight.data.size()
                    weight_params = torch.prod(torch.LongTensor(list(weight_size)))
                    params += weight_params.item()

                    if hasattr(module, "bias"):
                        try:
                            bias_size = module.bias.data.size()
                            bias_params = torch.prod(torch.LongTensor(list(bias_size)))
                            params += bias_params.item()
                        except:
                            pass

                flops = "Not Available"
                if class_name.find("Conv")!= -1 and hasattr(module,"weight") :
                    flops = (
                    torch.prod(torch.LongTensor(list(module.weight.data.size()))) * output.size(2) * output.size(
                        3)).item()

                elif isinstance(module, nn.Linear):
                   
                    flops = (torch.prod(torch.LongTensor(list(output.size()))) * input[0].size(1)).item()

                summary.append(
                    ModuleDetails(name=layer_name, input_size=list(input[0].size()), output_size=list(output.size()),
                                  num_parameters=params, multiply_adds=flops))

            if not isinstance(module, nn.ModuleList) and not isinstance(module, nn.Sequential) and module != self.model:
                hooks.append(module.register_forward_hook(hook))

        self.model.apply(add_hooks)

        space_len = 26

        self.model(input)
        for hook in hooks:
            hook.remove()

        details = "Model Summary" + os.linesep + "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
            ' ' * (space_len - len("Name")), ' ' * (space_len - len("Input Size")),
            ' ' * (space_len - len("Output Size")), ' ' * (space_len - len("Parameters")),
            ' ' * (space_len - len("Multiply Adds (Flops)"))) + os.linesep
        params_sum = 0
        flops_sum = 0
        for layer in summary:
            params_sum += layer.num_parameters
            if layer.multiply_adds != "Not Available":
                flops_sum += layer.multiply_adds
            details += "{}{}{}{}{}{}{}{}{}{}".format(layer.name, ' ' * (space_len - len(layer.name)), layer.input_size,
                                                     ' ' * (space_len - len(str(layer.input_size))), layer.output_size,
                                                     ' ' * (space_len - len(str(layer.output_size))),
                                                     layer.num_parameters,
                                                     ' ' * (space_len - len(str(layer.num_parameters))),
                                                     layer.multiply_adds,
                                                     ' ' * (space_len - len(str(layer.multiply_adds)))) + os.linesep

        details += os.linesep + "Total Parameters: {}".format(params_sum) + os.linesep
        details += "Total Multiply Adds (For Convolution aand Linear Layers only): {}".format(flops_sum) + os.linesep
        details += "Number of Layers" + os.linesep
        for layer in layer_instances:
            details += "{} : {} layers   ".format(layer,layer_instances[layer])

        return details

    """ Adds an hook that transforms the inputs before being passed into __train_func__
    
    The function should take in the input and return the transformed version
    Example:
        
        def transform_hook(input):
            
            return input * 0.5
    
    """
    def add_input_hook(self,function):
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
    def on_epoch_end(self, epoch,train_metrics,test_metrics, train_loss,duration):
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
    def on_training_completed(self, train_metrics,test_metrics,  train_duration):
        pass


""" Standard model suitable for training datasets represented as x and y pairs
    Arguments:
    model:  the module to be used for training, evaluation and inference.
    use_cuda_if_available: If set to true, training would be done on a gpu if any is available

"""
class StandardModel(BaseModel):
    def __init__(self,model,use_cuda_if_available=True):
        super(StandardModel,self).__init__(model,use_cuda_if_available)

    def __train_func__(self,data,optimizer,loss_fn,train_metrics,running_loss,epoch,batch_num):

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

    def __eval_function__(self,data,metrics):

        test_x, test_y = data
        if self.cuda:
            test_x = test_x.cuda()
            test_y = test_y.cuda()
        test_x = Variable(test_x)
        test_y = Variable(test_y)


        outputs = self.model(test_x)

        for metric in metrics:
            metric.update(outputs.cpu().data, test_y.cpu().data)

    def __predict_func__(self,inputs):

        if self.cuda:
            inputs = inputs.cuda()

        inputs = Variable(inputs)
        output = self.model(inputs)

        return output
