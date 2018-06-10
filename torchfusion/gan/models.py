import torch.nn as nn
from ..utils import *

import torch
from torch.autograd import Variable,grad
import torch.cuda as cuda
from torchvision import utils as vutils
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from ..utils import PlotInput,visualize
from torch.utils.data import DataLoader



"""This is the base Model for training, evaluating and performing inference with Generative Adversarial Networks
   Goodfellow et al. 2014 (https://arxiv.org/1406.2661)
All custom GAN trainers that use a single Generator and a single Discriminator should subclass this 

Arguments:
    gen_model:  the generator module.
    disc_model:  the discriminator module.
    use_cuda_if_available: If set to true, training would be done on a gpu if any is available"""

class BaseGANModel():
    def __init__(self, gen_model,disc_model,use_cuda_if_available=True):
        self.model_dir = os.getcwd()
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.cuda = False
        if use_cuda_if_available and cuda.is_available():
            self.cuda = True

        self.gen_loss_history = []
        self.disc_loss_history = []

        self.__input_hooks = []

    """Loads a pretrained generator model

        Arguments:
            path: a filepath or read-stream of an existing pre-trained model """

    def load_generator(self, path):
        checkpoint = torch.load(path)
        self.gen_model.load_state_dict(checkpoint)

    """Loads a pretrained discriminator model
            Arguments:
                path: a filepath or read-stream of an existing pre-trained model """

    def load_discriminator(self, path):
        checkpoint = torch.load(path)
        self.disc_model.load_state_dict(checkpoint)

    """ Saves the generator model
            Arguments:
                path: a filepath or write-stream to save the trained model"""
    def save_generator(self, path):
        torch.save(self.gen_model.state_dict(), path)

    """ Saves the discriminator model
            Arguments:
                path: a filepath or write-stream to save the trained model"""
    def save_discriminator(self, path):
        torch.save(self.disc_model.state_dict(), path)

    """trainer function for the Generator and Discriminator

                Arguments:
                    target: an instance of DataLoader containing the training set
                    source: an instance of DataLoader containing generator input
                    gen_optimizer: the optimizer for the generator
                    disc_optimizer: the optimizer for the discriminator
                    num_epochs: the maximum number of training epochs
                    disc_steps: number of times to train the discriminator before training the generator
                    gen_lr_schedule: An lr scheduler function that specifies the learning rate for each epoch
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
                    disc_lr_schedule: An lr scheduler function that specifies the learning rate for each epoch
                    Note: The function sheduler functions are called at the end of each epoch
                    model_dir: a path in which to save the models and the generated outputs
                    save_interval: Iteration interval at which to save generated outputs
                    notebook_mode: Optimizes the progress bar for either jupyter notebooks or consoles
                    batch_log: Enables printing of logs at every batch iteration
                    save_logs: Specifies a filepath in which to permanently save logs at every epoch
                    display_metrics: Enables display of metrics and loss visualizations at the end of each epoch.
                    save_metrics: Enables saving of metrics and loss visualizations at the end of each epoch.

                """
    def train(self, target,source,gen_optimizer,disc_optimizer,num_epochs=10, disc_steps=1, gen_lr_schedule=None,disc_lr_schedule=None, model_dir=os.getcwd(), save_interval=100,notebook_mode=False,batch_log=True,save_logs=None,display_metrics=True,save_metrics=True):
        assert(len(target.dataset) == len(source.dataset))
        assert(disc_steps < len(target.dataset))

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        self.model_dir = model_dir
        models_gen = os.path.join(model_dir, "gen_models")
        models_disc = os.path.join(model_dir, "disc_models")

        if not os.path.exists(models_gen):
            os.mkdir(models_gen)

        if not os.path.exists(models_disc):
            os.mkdir(models_disc)

        iterations = 0

        from tqdm import tqdm_notebook
        from tqdm import tqdm

        train_start_time = time()

        for e in tqdm(range(num_epochs)):

            self.gen_model.train()
            self.disc_model.train()
            self.on_epoch_start(e)

            running_gen_loss = torch.Tensor([0.0])
            running_disc_loss = torch.Tensor([0.0])
            gen_loss = 0.0
            disc_loss = 0.0
            gen_data_len = 0
            disc_data_len = 0

            if notebook_mode and batch_log:
                progress_ = tqdm_notebook(enumerate(zip(target,source)))
            elif batch_log:
                progress_ = tqdm(enumerate(zip(target,source)))
            else:
                progress_ = enumerate(zip(target,source))

            init_time = time()

            for i,(t,s) in progress_:

                if isinstance(t, list) or isinstance(t, tuple):
                    inputs = t[0]
                else:
                    inputs = t
                batch_size = inputs.size(0)
                disc_data_len += batch_size

                if len(self.__input_hooks) > 0:

                    for hook in self.__input_hooks:
                        inputs = hook(inputs)

                if isinstance(t, list):
                    t[0] = inputs
                elif isinstance(t, tuple):
                    t = (inputs,t[1])
                else:
                    t = inputs

                self.__disc_train_func__(t, s, disc_optimizer, running_disc_loss, e, i)

                disc_loss = running_disc_loss.data[0] / disc_data_len

                if (i+1) % disc_steps == 0:
                    self.__gen_train_func__(t, s, gen_optimizer, running_gen_loss, e, i)
                    gen_data_len += batch_size

                    gen_loss = running_gen_loss.data[0] / gen_data_len

                if batch_log:
                     progress_dict = {"Gen Loss": gen_loss,"Disc Loss":disc_loss}
                     progress_.set_postfix(progress_dict)

                iterations += 1

                if iterations % save_interval == 0:
                    self.save(s,iterations)
                    self.show(s,iterations)

                self.on_batch_end(e, i, gen_loss, disc_loss)
            if self.cuda:
                cuda.synchronize()
            duration = time() - init_time

            self.disc_loss_history.append(disc_loss)
            self.gen_loss_history.append(gen_loss)

            if gen_lr_schedule is not None:
                lr = gen_lr_schedule(e)
                adjust_learning_rate(lr,gen_optimizer)

            if disc_lr_schedule is not None:
                lr = disc_lr_schedule(e)
                adjust_learning_rate(lr, disc_optimizer)

            model_file = os.path.join(models_gen, "gen_model_{}.pth".format(e))
            self.save_generator(model_file)

            model_file = os.path.join(models_disc, "disc_model_{}.pth".format(e))
            self.save_discriminator(model_file)

            print("Epoch: {}, Duration: {} , Gen Loss: {} Disc Loss: {}".format(e, duration, gen_loss,disc_loss))

            if save_logs is not None:
                logfile = open(save_logs, "a")
                logfile.write("Epoch: {}, Duration: {} , Gen Loss: {} Disc Loss: {}".format(e, duration, gen_loss,disc_loss))
                logfile.close()

            epoch_arr = [x for x in range(e + 1)]

            if display_metrics or save_metrics:

                save_path = None

                if save_metrics:
                    save_path = os.path.join(model_dir, "epoch_{}_loss.png".format(e))

                visualize(epoch_arr, [PlotInput(value=self.gen_loss_history, name="Generator Loss", color="red"),
                                      PlotInput(value=self.disc_loss_history, name="Discriminator Loss", color="red")],display=display_metrics,
                          save_path=save_path)

            self.on_epoch_end(e,gen_loss, disc_loss, duration)
        train_end_time = time() - train_start_time
        self.on_training_completed(train_end_time)

    """ Abstract function containing the training logic for the discriminator, 
     all custom trainers must override this.

        Arguments:
            target: a single batch of data from the target set
            source: a single batch of data from the source set
            gen_optimizer: the optimizer for the generator
            running_loss: the current generator running loss
            epoch: current epoch
            batch_num: the current batch index
            """
    def __gen_train_func__(self,target,source,gen_optimizer,running_loss,epoch,batch_num):
        raise NotImplementedError()

    """ Abstract function containing the training logic for the generator, 
         all custom trainers must override this.

            Arguments:
                target: a single batch of data from the target set
                source: a single batch of data from the source set
                disc_optimizer: the optimizer for the discriminator
                running_loss: the current generator running loss
                epoch: current epoch
                batch_num: the current batch index
                """
    def __disc_train_func__(self,target,source,disc_optimizer,running_loss,epoch,batch_num):
        raise NotImplementedError()

    """ Abstract function containing logic to save the outputs of the generator, 
         all custom trainers must override this.

            Arguments:
                source: a single batch of data from the source set
                iterations: total number of batch iterations since the start of training
                """
    def save(self,source,iterations):
        raise NotImplementedError()

    """ Abstract function containing logic to display the outputs of the generator, 
             all custom trainers must override this.

                Arguments:
                    source: a single batch of data from the source set
                    iterations: total number of batch iterations since the start of training
                    """
    def show(self,source,iterations):
        raise NotImplementedError()

    """ Returns predictions for a given input tensor or a an instance of a DataLoader
            Arguments:
                inputs: input Tensor or DataLoader
    """
    def predict(self, inputs):
        self.gen_model.eval()

        if isinstance(inputs, DataLoader):
            predictions = []
            for i, data in enumerate(inputs):
                pred = self.__predict_func__(data)
                predictions.append(pred)

            output_array = []

            for batch in predictions:
                for pred in batch:
                    output_array.append(pred)
            return output_array
        else:
            pred = self.__predict_func__(inputs)

            return pred

    """ Abstract function containing custom logic for performing inference, must return the output,
            all custom trainers should implement this.
            input: An input tensor or a batch of input tensors
            """
    def __predict_func__(self, input):
        raise NotImplementedError()



    """ Callback that is invoked at the start of each epoch,
            epoch: the current epoch
            """
    def on_epoch_start(self, epoch):
        pass

    """ Callback that is invoked at the end of each epoch,
               epoch: the current epoch
               gen_loss: the running loss for the generator
               disc_loss: the running loss for the discriminator
               duration: Total duration of the current epoch
               """
    def on_epoch_end(self, epoch,gen_loss,disc_loss,duration):
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
                  gen_loss: the running loss for the generator
                  disc_loss: the running loss for the discriminator
                   """
    def on_batch_end(self, epoch, batch_num, gen_loss,disc_loss):
        pass

    """ Callback that is invoked at the end of training,
                  train_duration: the total duration of training
                   """
    def on_training_completed(self,train_duration):
        pass

    """ Returns a detailed summary of the generator, including input and output sizes, computational cost in flops,
        number of parameters and number of layers of each type
            input_size: The size of the input tensor
            """

    def gen_summary(self,input_size):

        input = torch.randn(input_size)

        input = input.unsqueeze(0)

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
                if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                    flops = (
                    torch.prod(torch.LongTensor(list(module.weight.data.size()))) * output.size(2) * output.size(
                        3)).item()

                if isinstance(module, nn.Linear):
                    flops = (torch.prod(torch.LongTensor(list(output.size()))) * input[0].size(1)).item()

                summary.append(
                    ModuleDetails(name=layer_name, input_size=list(input[0].size()), output_size=list(output.size()),
                                  num_parameters=params, multiply_adds=flops))

            if not isinstance(module, nn.ModuleList) and not isinstance(module, nn.Sequential) and module != self.gen_model:
                hooks.append(module.register_forward_hook(hook))

        self.gen_model.apply(add_hooks)

        space_len = len("                             ")

        self.gen_model(input)
        for hook in hooks:
            hook.remove()

        details = "Generator Summary " + os.linesep + "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
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
            details += "{} : {} layers   ".format(layer, layer_instances[layer])

        return details

    """ Returns a detailed summary of the discriminator, including input and output sizes, computational cost in flops,
            number of parameters and number of layers of each type
                input_size: The size of the input tensor
                """
    def disc_summary(self,input_size):

        input = torch.randn(input_size)

        input = input.unsqueeze(0)

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
                if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                    flops = (
                    torch.prod(torch.LongTensor(list(module.weight.data.size()))) * output.size(2) * output.size(
                        3)).item()

                if isinstance(module, nn.Linear):
                    flops = (torch.prod(torch.LongTensor(list(output.size()))) * input[0].size(1)).item()

                summary.append(
                    ModuleDetails(name=layer_name, input_size=list(input[0].size()), output_size=list(output.size()),
                                  num_parameters=params, multiply_adds=flops))

            if not isinstance(module, nn.ModuleList) and not isinstance(module, nn.Sequential) and module != self.disc_model:
                hooks.append(module.register_forward_hook(hook))

        self.disc_model.apply(add_hooks)

        space_len = 26

        self.disc_model(input)
        for hook in hooks:
            hook.remove()

        details = "Discriminator Summary " + os.linesep + "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
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
            details += "{} : {} layers    ".format(layer, layer_instances[layer])

        return details

    """ Adds an hook that transforms the inputs before being passed into __train_func__

        The function should take in the input and return the transformed version
        Example:

            def transform_hook(input):

                return input * 0.5

        """

    def add_input_hook(self, function):
        self.__input_hooks.append(function)

""" StandardGANModel model suitable for unconditional GANS with an explicit loss function
    Arguments:
    gen_model:  the generator module.
    disc_model:  the discriminator module.
    gen_loss_fn: the loss function to be used for the generator
    disc_loss_fn: the loss function to be used for the discriminator
    smooth_labels: if True, real labels range between 0.7 - 1.2 and fake labels range between 0.0 - 0.3
    use_cuda_if_available: If set to true, training would be done on a gpu if any is available

"""
class StandardGANModel(BaseGANModel):
    def __init__(self, gen_model, disc_model, gen_loss_fn,disc_loss_fn,smooth_labels=False,use_cuda_if_available=True):
        super(StandardGANModel, self).__init__(gen_model, disc_model, use_cuda_if_available)
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.smooth_labels = smooth_labels

    def __disc_train_func__(self, target, source, optimizer,running_loss, epoch, batch_num):

        for params in self.disc_model.parameters():
            params.requires_grad = True

        optimizer.zero_grad()

        if isinstance(target, list) or isinstance(target, tuple):
            x = target[0]
        else:
            x = target

        batch_size = x.size(0)

        if self.smooth_labels:
            real_labels = torch.randn(batch_size).uniform_(0.7,1.2)
            fake_labels = torch.randn(batch_size).uniform_(0.0,0.3)
        else:
            real_labels = torch.ones(batch_size)
            fake_labels = torch.zeros(batch_size)

        if self.cuda:
            x = x.cuda()
            source = source.cuda()
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()

        x = Variable(x)
        source = Variable(source)
        real_labels = Variable(real_labels)
        fake_labels = Variable(fake_labels)

        outputs = self.disc_model(x)

        real_loss = self.disc_loss_fn(outputs, real_labels)
        real_loss.backward()

        generated = self.gen_model(source)
        gen_outputs = self.disc_model(generated.detach())

        fake_loss = self.disc_loss_fn(gen_outputs, fake_labels)
        fake_loss.backward()

        optimizer.step()

        d_loss = real_loss + fake_loss

        running_loss.add_(d_loss.cpu() * batch_size)

    def __gen_train_func__(self, target, source, optimizer, running_loss, epoch, batch_num):

        for params in self.disc_model.parameters():
            params.requires_grad = False

        optimizer.zero_grad()

        if isinstance(target, list) or isinstance(target, tuple):
            x = target[0]
        else:
            x = target
        batch_size = x.size(0)
        labels = torch.ones(batch_size)
        if self.smooth_labels:
            labels = torch.randn(batch_size).uniform_(0.7,1.2)

        if self.cuda:
            source = source.cuda()
            labels = labels.cuda()

        source = Variable(source)
        labels = Variable(labels)

        fake_images = self.gen_model(source)
        outputs = self.disc_model(fake_images)

        loss = self.gen_loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss.add_(loss.cpu() * batch_size)

    def save(self, source, iteration):

        save_dir = os.path.join(self.model_dir, "gen_images")

        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)
        images_file = os.path.join(save_dir, "image_{}.png".format(iteration))

        if self.cuda:
            source = source.cuda()

        source = Variable(source)
        outputs = self.gen_model(source)
        vutils.save_image(outputs.cpu().data, images_file, normalize=True)

    def show(self, source, iteration):

        if self.cuda:
            source = source.cuda()

        source = Variable(source)
        outputs = self.gen_model(source)

        images = vutils.make_grid(outputs.cpu().data, normalize=True)

        images = np.transpose(images.numpy(), (1, 2, 0))
        plt.imshow(images)
        plt.show()

    def __predict_func__(self, source):

        source = Variable(source)
        if self.cuda:
            source = source.cuda()

        source = Variable(source)
        outputs = self.gen_model(source)

        return outputs

""" StandardGANModel model suitable for unconditional GANS that use the Improved Wasserstein Distance
    See Gulrajani et al. 2017 (https://arxiv.org/1704.00028)
    based on earlier work by Arjovsky et al. 2017 (https://arxiv.org/1701.07875)
    Arguments:
        gen_model:  the generator module.
        disc_model:  the discriminator module.
        lambda_: Lambda parameter to use for the gradient penalty.
        use_cuda_if_available: If set to true, training would be done on a gpu if any is available

"""
class WGANModel(BaseGANModel):
    def __init__(self, gen_model, disc_model,lambda_ = 0.25, use_cuda_if_available=True):
        super(WGANModel, self).__init__(gen_model, disc_model, use_cuda_if_available)
        self.lambda_ = lambda_

    def __disc_train_func__(self, target, source, disc_optimizer, running_loss, epoch, batch_num):

        for params in self.disc_model.parameters():
            params.requires_grad = True

        disc_optimizer.zero_grad()

        if isinstance(target, list) or isinstance(target, tuple):
            x = target[0]
        else:
            x = target
        batch_size = x.size(0)

        if self.cuda:
            x = x.cuda()
            source = source.cuda()

        x = Variable(x)
        source = Variable(source)

        real_loss = -torch.mean(self.disc_model(x))
        real_loss.backward()

        generated = self.gen_model(source).detach()

        gen_loss = torch.mean(self.disc_model(generated))
        gen_loss.backward()

        eps = torch.randn(x.size()).uniform_(0,1)

        if self.cuda:
            eps = eps.cuda()

        x__ = Variable(eps * x.data + (1.0 - eps) * generated.data,requires_grad=True)

        pred__ = self.disc_model(x__)

        grad_outputs = torch.ones(pred__.size())

        if self.cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = grad(outputs=pred__,inputs=x__,grad_outputs=grad_outputs,create_graph=True,retain_graph=True,only_inputs=True)[0]

        gradient_penalty = self.lambda_ * ((gradients.view(gradients.size(0),-1).norm(2,1) - 1) ** 2).mean()

        gradient_penalty.backward()

        loss = real_loss + gen_loss + gradient_penalty

        disc_optimizer.step()

        running_loss.add_(loss.cpu() * batch_size)

    def __gen_train_func__(self, target, source, optimizer, running_loss, epoch, batch_num):


        for params in self.disc_model.parameters():
            params.requires_grad = False

        optimizer.zero_grad()

        if isinstance(target,list) or isinstance(target,tuple):
            x = target[0]
        else:
            x = target
        batch_size = x.size(0)

        if self.cuda:
            source = source.cuda()

        source = Variable(source)

        fake_images = self.gen_model(source)
        loss = -torch.mean(self.disc_model(fake_images))

        loss.backward()

        optimizer.step()

        running_loss.add_(loss.cpu() * batch_size)

    def save(self, source, iteration):

        save_dir = os.path.join(self.model_dir, "gen_images")

        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)
        images_file = os.path.join(save_dir, "image_{}.png".format(iteration))

        if self.cuda:
            source = source.cuda()

        source = Variable(source)
        outputs = self.gen_model(source)
        vutils.save_image(outputs.cpu().data, images_file, normalize=True)

    def show(self, source, iteration):

        if self.cuda:
            source = source.cuda()

        source = Variable(source)
        outputs = self.gen_model(source)

        images = vutils.make_grid(outputs.cpu().data, normalize=True)

        images = np.transpose(images.numpy(), (1, 2, 0))
        plt.imshow(images)
        plt.show()

    def __predict_func__(self, input):
        source = Variable(input)
        if self.cuda:
            source = source.cuda()

        source = Variable(source)
        outputs = self.gen_model(source)

        return outputs




