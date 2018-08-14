from collections import namedtuple
import torch.nn as nn
from torchfusion.utils import *
from torchfusion.learners import AbstractBaseLearner
from torchfusion.gan.utils import ImagePool

import torch
from torch.autograd import Variable, grad
import torch.cuda as cuda
from torchvision import utils as vutils
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from torchfusion.utils import PlotInput, visualize
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.distributions as distribution
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.onnx as onnx
from math import ceil



r"""This is the base Learner for training, evaluating and performing inference with Generative Adversarial Networks
Goodfellow et al. 2014 (https://arxiv.org/1406.2661)
All custom GAN learners that use a single Generator and a single Discriminator should subclass this 

    Args:
        gen_model (Module):  the generator module.
        disc_model (Module):  the discriminator module.
        use_cuda_if_available (boolean): If set to true, training would be done on a gpu if any is available"""

class BaseGanLearner(AbstractBaseLearner):
    def __init__(self, gen_model,disc_model,use_cuda_if_available=True):
        super(BaseGanLearner,self).__init__()
        self.model_dir = os.getcwd()
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.cuda = False
        if use_cuda_if_available and cuda.is_available():
            self.cuda = True

        self.gen_loss_history = []
        self.disc_loss_history = []

        self.gen_optimizer = None
        self.disc_optimizer = None
        self.gen_running_loss = None
        self.disc_running_loss = None

        self.visdom_log = None
        self.tensorboard_log = None

    r"""Initialize generator model weights using pre-trained weights from the filepath

        Args:
            path (str): path to a compatible pre-defined model

        """
    def load_generator(self, path):
        checkpoint = torch.load(path)
        self.gen_model.load_state_dict(checkpoint)

    r"""Initialize discriminator model weights using pre-trained weights from the filepath

        Args:
            path (str): path to a compatible pre-defined model

        """

    def load_discriminator(self, path):
        checkpoint = torch.load(path)
        self.disc_model.load_state_dict(checkpoint)

    r"""Saves the generator model to the path specified
            Args:
                path (str): path to save model
                save_architecture (boolean): if True, both weights and architecture will be saved, default is False

            """
    def save_generator(self, path,save_architecture=False):
        if save_architecture:
            torch.save(self.gen_model,path)
        else:
            torch.save(self.gen_model.state_dict(), path)

    r"""Saves the discriminator model to the path specified
            Args:
                path (str): path to save model
                save_architecture (boolean): if True, both weights and architecture will be saved, default is False

            """
    def save_discriminator(self, path,save_architecture=False):
        if save_architecture:
            torch.save(self.disc_model,path)
        else:
            torch.save(self.disc_model.state_dict(), path)

    r"""Complete Training loop
    
            Args:
                train_loader (DataLoader): an instance of DataLoader containing the training set
                gen_optimizer (Optimizer): an optimizer for updating parameters of the generator
                disc_optimizer (Optimizer): an optimizer for updating parameters of the discriminator
                num_epochs (int): The maximum number of training epochs
                disc_steps (int): The number of times to train the discriminator before training generator
                save_models (str): If all, the model is saved at the end of each epoch while the best models based 
                gen_lr_scheduler (_LRScheduler): Learning rate scheduler for the generator
                disc_lr_scheduler (_LRScheduler): Learning rate sheduler for the discriminator
                    on the test set are also saved in best_models folder
                    If 'best', only the best models are saved,  test_loader and test_metrics must be provided
                model_dir (str) = a path in which to save the models
                save_interval (int): Saves sample generated outputs at the the number of iterations specified
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

    def __train_loop__(self, train_loader,gen_optimizer,disc_optimizer,num_epochs=10, disc_steps=1,gen_lr_scheduler=None,disc_lr_scheduler=None, model_dir=os.getcwd(),save_model_interval=1, save_outputs_interval=100,display_outputs=True,notebook_mode=False,batch_log=True,save_logs=None,display_metrics=False,save_metrics=False,visdom_log=None,tensorboard_log=None,save_architecture=False):
        assert(disc_steps < len(train_loader.dataset))

        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.tensorboard_log = tensorboard_log
        self.visdom_log = visdom_log

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

        for e in range(num_epochs):
            print("Epoch {} of {}".format(e + 1, num_epochs))

            self.gen_model.train()
            self.disc_model.train()
            for func in self.epoch_start_funcs:
                func(e + 1)

            self.gen_running_loss = torch.Tensor([0.0])
            self.disc_running_loss = torch.Tensor([0.0])
            gen_loss = 0
            disc_loss = 0

            gen_data_len = 0
            disc_data_len = 0

            if notebook_mode and batch_log:
                progress_ = tqdm_notebook(enumerate(train_loader))
            elif batch_log:
                progress_ = tqdm(enumerate(train_loader))
            else:
                progress_ = enumerate(train_loader)

            max_batch_size = 0

            init_time = time()

            for i,t in progress_:

                for func in self.batch_start_funcs:
                    func(e + 1,i + 1)

                if isinstance(t, list) or isinstance(t, tuple):
                    inputs = t[0]
                else:
                    inputs = t
                batch_size = inputs.size(0)
                disc_data_len += batch_size

                if max_batch_size < batch_size:
                    max_batch_size = batch_size

                self.__disc_train_func__(t)

                disc_loss = self.disc_running_loss.data.item() / disc_data_len

                if (i+1) % disc_steps == 0:
                    self.__gen_train_func__(t)
                    gen_data_len += batch_size

                    gen_loss = self.gen_running_loss.data.item() / gen_data_len

                if batch_log:
                     progress_dict = {"Gen Loss": gen_loss,"Disc Loss":disc_loss}
                     progress_.set_postfix(progress_dict)

                iterations += 1

                if iterations % save_outputs_interval == 0:
                    self.__save__(iterations)
                    if display_outputs:
                        self.__show__(iterations)

                if batch_log:

                    progress_.set_description("{}/{} batches ".format(int(ceil(disc_data_len / max_batch_size)),
                                                                      int(ceil(len(
                                                                          train_loader.dataset) / max_batch_size))))
                    progress_dict = {"Disc Loss": disc_loss,"Gen Loss":gen_loss}

                    progress_.set_postfix(progress_dict)

                batch_info = {"gen_loss":gen_loss , "disc_loss": disc_loss}

                for func in self.batch_end_funcs:
                    func(e + 1,i + 1,batch_info)


            if self.cuda:
                cuda.synchronize()
            duration = time() - init_time

            if gen_lr_scheduler is not None:
                if isinstance(gen_lr_scheduler, ReduceLROnPlateau):
                    gen_lr_scheduler.step(gen_loss)
                else:
                    gen_lr_scheduler.step()

            if disc_lr_scheduler is not None:
                if isinstance(disc_lr_scheduler, ReduceLROnPlateau):
                    disc_lr_scheduler.step(gen_loss)
                else:
                    disc_lr_scheduler.step()

            self.disc_loss_history.append(disc_loss)
            self.gen_loss_history.append(gen_loss)


   
            if (e+1) % save_model_interval == 0:

                model_file = os.path.join(models_gen, "gen_model_{}.pth".format(e + 1))
                self.save_generator(model_file, save_architecture)

                print("New Generator model saved at {}".format(model_file))

                model_file = os.path.join(models_disc, "disc_model_{}.pth".format(e + 1))
                self.save_discriminator(model_file, save_architecture)

                print("New Discriminator model saved at {}".format(model_file))


            print("Epoch: {}, Duration: {} , Gen Loss: {} Disc Loss: {}".format(e+1, duration, gen_loss,disc_loss))

            if save_logs is not None:
                logfile = open(save_logs, "a")
                logfile.write("Epoch: {}, Duration: {} , Gen Loss: {} Disc Loss: {}".format(e+1, duration, gen_loss,disc_loss))
                logfile.close()

            epoch_arr = [x for x in range(e + 1)]
            epoch_arr_tensor = torch.LongTensor(epoch_arr)

            if display_metrics or save_metrics:

                save_path = None

                if save_metrics:
                    save_path = os.path.join(model_dir, "epoch_{}_loss.png".format(e+1))

                visualize(epoch_arr, [PlotInput(value=self.gen_loss_history, name="Generator Loss", color="red"),
                                      PlotInput(value=self.disc_loss_history, name="Discriminator Loss", color="red")],display=display_metrics,
                          save_path=save_path)

            batch_info = {"gen_loss": gen_loss, "disc_loss": disc_loss, "duration": duration}
            for func in self.epoch_end_funcs:
                func(e + 1, batch_info)



            if visdom_log is not None:
                visdom_log.plot_line(torch.FloatTensor(self.gen_loss_history), epoch_arr_tensor, win="gen_loss",
                                     title="Generator Loss")

                visdom_log.plot_line(torch.FloatTensor(self.disc_loss_history), epoch_arr_tensor, win="disc_loss",
                                     title="Discriminator Loss")

            if tensorboard_log is not None:
                writer = SummaryWriter(os.path.join(model_dir, tensorboard_log))
                for epoch in epoch_arr:
                    writer.add_scalar("logs/gen_loss", self.gen_loss_history[epoch], global_step=epoch)
                    writer.add_scalar("logs/disc_loss", self.disc_loss_history[epoch], global_step=epoch)

                writer.close()

            batch_info = {"gen_loss": gen_loss, "disc_loss": disc_loss, "duration": duration}
            for func in self.epoch_end_funcs:
                func(e + 1, batch_info)

        train_end_time = time() - train_start_time
        batch_info = {"train_duration": train_end_time}
        for func in self.train_completed_funcs:
            func(batch_info)

    """ Abstract function containing the training logic for the generator, 
     all custom trainers must override this.

        Arguments:
            target: a single batch of data from the target set
            source: a single batch of data from the source set
            gen_optimizer: the optimizer for the generator
            running_loss: the current generator running loss
            epoch: current epoch
            batch_num: the current batch index
            """
    def __gen_train_func__(self,data):
        raise NotImplementedError()

    """ Abstract function containing the training logic for the discriminator, 
         all custom trainers must override this.

            Arguments:
                target: a single batch of data from the target set
                source: a single batch of data from the source set
                disc_optimizer: the optimizer for the discriminator
                running_loss: the current generator running loss
                epoch: current epoch
                batch_num: the current batch index
                """
    def __disc_train_func__(self,data):
        raise NotImplementedError()

    """ Abstract function containing logic to save the outputs of the generator, 
         all custom trainers must override this.

            Arguments:
                source: a single batch of data from the source set
                iterations: total number of batch iterations since the start of training
                """
    def __save__(self,iterations):
        raise NotImplementedError()

    """ Abstract function containing logic to display the outputs of the generator, 
             all custom trainers must override this.

                Arguments:
                    source: a single batch of data from the source set
                    iterations: total number of batch iterations since the start of training
                    """

    def  evaluate(self, *args):
        pass
    def validate(self, *args):
        pass

    def __show__(self,iterations):
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
                batch_pred = self.__predict_func__(data)
                for pred in batch_pred:
                    predictions.append(pred)
            return predictions

        else:
            pred = self.__predict_func__(inputs)

            return pred

    """ Abstract function containing custom logic for performing inference, must return the output,
            all custom trainers should implement this.
            input: An input tensor or a batch of input tensors
            """
    def __predict_func__(self, input):
        raise NotImplementedError()


class StandardGanCore(BaseGanLearner):
    def __init__(self, gen_model, disc_model, use_cuda_if_available=True):
        super().__init__(gen_model, disc_model, use_cuda_if_available)

        self.latent_size = None
        self.dist = None
        self.fixed_source = None

        self.conditional = None
        self.classes = None
        self.num_samples = None

    def __init_train_params__(self, latent_size, dist, num_classes, num_samples):
        self.latent_size = latent_size
        self.dist = dist
        self.classes = num_classes
        self.num_samples = num_samples
        self.conditional = (num_classes is not None)

    def __disc_train_func__(self, data):

        for params in self.disc_model.parameters():
            params.requires_grad = True

        for params in self.gen_model.parameters():
            params.requires_grad = False

        if isinstance(data, list) or isinstance(data, tuple):
            x = data[0]

        else:
            if self.conditional:
                raise ValueError("Conditional mode is invalid for inputs with no labels, set num_classes to None or provide labels")
            x = data

        batch_size = x.size(0)
        if isinstance(self.latent_size, int):
            source_size = list([self.latent_size])
        else:
            source_size = list(self.latent_size)
        cond_samples_size = [self.num_samples] + source_size
        source_size = [batch_size] + source_size

        if self.fixed_source is None:

            if self.conditional:
                self.fixed_source = self.dist.sample(tuple(cond_samples_size))

            else:
                self.fixed_source = self.dist.sample(tuple(source_size))

            if self.cuda:
                self.fixed_source = self.fixed_source.cuda()

            self.fixed_source = Variable(self.fixed_source)


    def __gen_train_func__(self, data):

        for params in self.gen_model.parameters():
            params.requires_grad = True

        for params in self.disc_model.parameters():
            params.requires_grad = False


    def __save__(self, iteration):

        save_dir = os.path.join(self.model_dir, "gen_images")

        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)
        if self.tensorboard_log is not None:
            writer = SummaryWriter(self.tensorboard_log)

        if self.conditional:

            for i in range(self.classes):
                class_path = os.path.join(save_dir,"class_{}".format(i))
                if not os.path.exists(class_path):
                    os.mkdir(class_path)
                class_labels = torch.randn((self.num_samples, 1)).type(torch.LongTensor).fill_(i)

                if self.cuda:
                    class_labels = class_labels.cuda()

                class_labels = Variable(class_labels)
                outputs = self.gen_model(self.fixed_source, class_labels)

                images_file = os.path.join(class_path, "iteration{}.png".format(iteration))

                images = vutils.make_grid(outputs.cpu().data, normalize=True)
                vutils.save_image(outputs.cpu().data, images_file, normalize=True)

                if self.tensorboard_log is not None:
                    writer.add_image("logs/gen_images/class_{}".format(i), images, global_step=iteration)

                if self.visdom_log is not None:
                    self.visdom_log.log_images(images, win="class_{}".format(i), title="Class {}".format(i))

        else:
            outputs = self.gen_model(self.fixed_source)
            images_file = os.path.join(save_dir, "image_{}.png".format(iteration))

            images = vutils.make_grid(outputs.cpu().data, normalize=True)
            vutils.save_image(outputs.cpu().data, images_file, normalize=True)

            if self.tensorboard_log is not None:
                writer.add_image("logs/gen_images", images, global_step=iteration)

            if self.visdom_log is not None:
                self.visdom_log.log_images(images, win="gen_images", title="Generated Images")
        if self.tensorboard_log:
            writer.close()

    def __show__(self, iteration):

        if self.conditional:
            for i in range(self.classes):
                class_labels = torch.randn((self.num_samples, 1)).type(torch.LongTensor).fill_(i)

                if self.cuda:
                    class_labels = class_labels.cuda()

                class_labels = Variable(class_labels)
                outputs = self.gen_model(self.fixed_source, class_labels)

                images = vutils.make_grid(outputs.cpu().data, normalize=True)
                images = np.transpose(images.numpy(), (1, 2, 0))
                plt.subplot(self.classes, 1, i + 1)
                plt.axis("off")
                # plt.title("class {}".format(i))
                plt.imshow(images)

            plt.show()

        else:
            outputs = self.gen_model(self.fixed_source)

            images = vutils.make_grid(outputs.cpu().data, normalize=True)

            images = np.transpose(images.numpy(), (1, 2, 0))
            plt.imshow(images)
            plt.axis("off")
            plt.grid(False)
            plt.show()

    def __predict_func__(self, data):

        if isinstance(data, list) or isinstance(data, tuple):
            source = data[0]
            if self.conditional:
                labels = data[1]
        else:
            source = data

        if self.cuda:
            source = source.cuda()
            if self.conditional:
                labels = labels.cuda()

        source = Variable(source)

        if self.conditional:
            labels = Variable(labels.unsqueeze(1))
            outputs = self.gen_model(source, labels)
        else:
            outputs = self.gen_model(source)

        return outputs

    def gen_summary(self, input_size, label=None, input_type=torch.FloatTensor, item_length=26, tensorboard_log=None):
        input = torch.randn(input_size).type(input_type).unsqueeze(0)
        if label is None:
            inputs = Variable(input.cuda if self.cuda else input)
        else:
            label = torch.randn(1, 1).fill_(label).long()
            inputs = [Variable(input.cuda() if self.cuda else input), Variable(label.cuda() if self.cuda else label)]

        return get_model_summary(self.gen_model, inputs, item_length=item_length, tensorboard_log=tensorboard_log)

    def disc_summary(self, input_size, label=None, input_type=torch.FloatTensor, item_length=26, tensorboard_log=None):
        input = torch.randn(input_size).type(input_type).unsqueeze(0)
        if label is None:
            inputs = Variable(input.cuda if self.cuda else input)
        else:
            label = torch.randn(1, 1).fill_(label).long()
            inputs = [Variable(input.cuda() if self.cuda else input), Variable(label.cuda() if self.cuda else label)]

        return get_model_summary(self.disc_model, inputs, item_length=item_length, tensorboard_log=tensorboard_log)

    def gen_to_onnx(self, path, input_size, label=None, input_type=torch.FloatTensor, **kwargs):
        input = torch.randn(input_size).type(input_type).unsqueeze(0)
        if label is None:
            inputs = Variable(input.cuda() if self.cuda else input)
        else:
            label = torch.randn(1, 1).fill_(label)
            inputs = [Variable(input.cuda() if self.cuda else input), label.cuda() if self.cuda else label]

        return onnx._export(self.gen_model, inputs, f=path, **kwargs)

    def disc_to_onnx(self, path, input_size, label=None, input_type=torch.FloatTensor, **kwargs):
        input = torch.randn(input_size).type(input_type).unsqueeze(0)
        if label is None:
            inputs = Variable(input.cuda() if self.cuda else input)
        else:
            label = torch.randn(1, 1).fill_(label)
            inputs = [Variable(input.cuda() if self.cuda else input), label.cuda() if self.cuda else label]

        return onnx._export(self.disc_model, inputs, f=path, **kwargs)


class StandardBaseGanLearner(StandardGanCore):
    def __init__(self, gen_model, disc_model, use_cuda_if_available=True):
        super().__init__(gen_model, disc_model, use_cuda_if_available)

        self.latent_size = None
        self.dist = None
        self.fixed_source = None

        self.conditional = None
        self.classes = None
        self.num_samples = None
        self.relative_mode = None

    def __init_train_params__(self,latent_size,dist,num_classes,num_samples,relative_mode):
        self.latent_size = latent_size
        self.dist = dist
        self.classes = num_classes
        self.num_samples = num_samples
        self.conditional = (num_classes is not None)
        self.relative_mode = relative_mode

    def __disc_train_func__(self, data):

        super().__disc_train_func__(data)

        self.disc_optimizer.zero_grad()

        if isinstance(data, list) or isinstance(data, tuple):
            x = data[0]
            if self.conditional:
                class_labels = data[1]
        else:
            x = data

        batch_size = x.size(0)
        if isinstance(self.latent_size, int):
            source_size = list([self.latent_size])
        else:
            source_size = list(self.latent_size)
        source_size = [batch_size] + source_size

        source = self.dist.sample(tuple(source_size))

        if self.cuda:
            x = x.cuda()
            source = source.cuda()

            if self.conditional:
                class_labels = class_labels.cuda()

        x = Variable(x)
        source = Variable(source)
        if self.conditional:
            class_labels = Variable(class_labels)

        if self.conditional:
            outputs = self.disc_model(x, class_labels.unsqueeze(1))
        else:
            outputs = self.disc_model(x)

        if self.conditional:
            random_labels = torch.from_numpy(np.random.randint(0, self.classes, size=(batch_size, 1))).long()
            if self.cuda:
                random_labels = random_labels.cuda()

            random_labels = Variable(random_labels)

            generated = self.gen_model(source, random_labels)
            gen_outputs = self.disc_model(generated.detach(), random_labels)

        else:
            generated = self.gen_model(source)
            gen_outputs = self.disc_model(generated.detach())

        loss = self.__update_discriminator_loss__(x,generated,outputs,gen_outputs)
        loss.backward()
        self.disc_optimizer.step()

        self.disc_running_loss.add_(loss.cpu() * batch_size)

    def __gen_train_func__(self, data):

        super().__gen_train_func__(data)

        self.gen_optimizer.zero_grad()

        if isinstance(data, list) or isinstance(data, tuple):
            x = data[0]
            if self.conditional and self.relative_mode:
                class_labels = data[1].unsqueeze(1)
        else:
            x = data
        batch_size = x.size(0)

        if isinstance(self.latent_size, int):
            source_size = list([self.latent_size])
        else:
            source_size = list(self.latent_size)

        source_size = [batch_size] + source_size

        source = self.dist.sample(tuple(source_size))
        if self.conditional:
            random_labels = torch.from_numpy(np.random.randint(0, self.classes, size=(batch_size, 1))).long()
            if self.cuda:
                random_labels = random_labels.cuda()

            random_labels = Variable(random_labels)

        if self.cuda:
            source = source.cuda()

            x = x.cuda()
            if self.conditional and self.relative_mode:
                class_labels = class_labels.cuda()

        source = Variable(source)
        x = Variable(x)
        if self.conditional and self.relative_mode:
            class_labels = Variable(class_labels)

        if self.conditional:
            fake_images = self.gen_model(source, random_labels)
            outputs = self.disc_model(fake_images, random_labels)
            if self.relative_mode:
                real_outputs = self.disc_model(x, class_labels)
        else:
            fake_images = self.gen_model(source)
            outputs = self.disc_model(fake_images)
            if self.relative_mode:
                real_outputs = self.disc_model(x)

        if not self.relative_mode:
            real_outputs = None


        loss = self.__update_generator_loss__(x,fake_images,real_outputs,outputs)
        loss.backward()

        self.gen_optimizer.step()

        self.gen_running_loss.add_(loss.cpu() * batch_size)

    def __update_generator_loss__(self,real_images,gen_images,real_preds,gen_preds):
        raise NotImplementedError()

    def __update_discriminator_loss__(self,real_images,gen_images,real_preds,gen_preds):
        raise NotImplementedError()

class StandardGanLearner(StandardBaseGanLearner):
    def __init__(self, gen_model, disc_model, use_cuda_if_available=True):
        super(StandardGanLearner, self).__init__(gen_model, disc_model, use_cuda_if_available)

    def train(self,train_loader, gen_optimizer,disc_optimizer,latent_size,gen_loss_fn=nn.BCELoss(), disc_loss_fn=nn.BCELoss(),num_epochs=10, disc_steps=1,gen_lr_scheduler=None,disc_lr_scheduler=None, model_dir=os.getcwd(), save_model_interval=1, save_outputs_interval=100,display_outputs=True,notebook_mode=False,batch_log=True,save_logs=None,display_metrics=True,save_metrics=True,visdom_log=None,tensorboard_log=None,save_architecture=False,dist=distribution.Normal(0,1),
                 real_labels=1,fake_labels=0, num_classes=None, num_samples=5):

        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.real_labels = real_labels
        self.fake_labels = fake_labels

        self.__init_train_params__(latent_size,dist,num_classes,num_samples,False)


        super().__train_loop__(train_loader,gen_optimizer,disc_optimizer,num_epochs, disc_steps,gen_lr_scheduler,disc_lr_scheduler, model_dir, save_model_interval, save_outputs_interval,display_outputs,notebook_mode,batch_log,save_logs,display_metrics,save_metrics,visdom_log,tensorboard_log,save_architecture)


    def __update_discriminator_loss__(self,x,gen_images,real_preds,gen_preds):

        batch_size = x.size(0)
        real_labels = torch.randn(batch_size, 1).fill_(self.real_labels)
        fake_labels = torch.randn(batch_size, 1).fill_(self.fake_labels)

        real_labels = Variable(real_labels.cuda() if self.cuda else real_labels)
        fake_labels = Variable(fake_labels.cuda() if self.cuda else fake_labels)

        real_loss = self.disc_loss_fn(real_preds,real_labels)

        gen_loss = self.gen_loss_fn(gen_preds,fake_labels)


        return real_loss + gen_loss

    def __update_generator_loss__(self,x,gen_images,real_preds,gen_preds):
        batch_size = x.size(0)
        real_labels = torch.ones(batch_size, 1)

        real_labels = Variable(real_labels.cuda() if self.cuda else real_labels)

        loss = self.gen_loss_fn(gen_preds,real_labels)

        return loss

class RStandardGanLearner(StandardBaseGanLearner):
    def __init__(self, gen_model, disc_model, use_cuda_if_available=True):
        super(RStandardGanLearner, self).__init__(gen_model, disc_model, use_cuda_if_available)

    def train(self,train_loader, gen_optimizer,disc_optimizer,latent_size,loss_fn=nn.BCEWithLogitsLoss(),num_epochs=10, disc_steps=1,gen_lr_scheduler=None,disc_lr_scheduler=None, model_dir=os.getcwd(), save_model_interval=1, save_outputs_interval=100,display_outputs=True,notebook_mode=False,batch_log=True,save_logs=None,display_metrics=True,save_metrics=True,visdom_log=None,tensorboard_log=None,save_architecture=False,dist=distribution.Normal(0,1),
                  num_classes=None, num_samples=5):

        self.loss_fn = loss_fn


        self.__init_train_params__(latent_size,dist,num_classes,num_samples,True)


        super().__train_loop__(train_loader,gen_optimizer,disc_optimizer,num_epochs, disc_steps,gen_lr_scheduler,disc_lr_scheduler, model_dir, save_model_interval, save_outputs_interval,display_outputs,notebook_mode,batch_log,save_logs,display_metrics,save_metrics,visdom_log,tensorboard_log,save_architecture)


    def __update_discriminator_loss__(self,x,gen_images,real_preds,gen_preds):

        batch_size = x.size(0)
        real_labels = torch.ones(batch_size, 1)

        real_labels = Variable(real_labels.cuda() if self.cuda else real_labels)

        loss = self.loss_fn(real_preds - gen_preds,real_labels)

        return loss

    def __update_generator_loss__(self,x,gen_images,real_preds,gen_preds):
        batch_size = x.size(0)
        real_labels = torch.ones(batch_size, 1)

        real_labels = Variable(real_labels.cuda() if self.cuda else real_labels)

        loss = self.loss_fn(gen_preds - real_preds,real_labels)

        return loss


class RAvgStandardGanLearner(StandardBaseGanLearner):
    def __init__(self, gen_model, disc_model, use_cuda_if_available=True):
        super(RAvgStandardGanLearner, self).__init__(gen_model, disc_model, use_cuda_if_available)

    def train(self,train_loader, gen_optimizer,disc_optimizer,latent_size,loss_fn=nn.BCEWithLogitsLoss(),num_epochs=10, disc_steps=1,gen_lr_scheduler=None,disc_lr_scheduler=None, model_dir=os.getcwd(), save_model_interval=1, save_outputs_interval=100,display_outputs=True,notebook_mode=False,batch_log=True,save_logs=None,display_metrics=True,save_metrics=True,visdom_log=None,tensorboard_log=None,save_architecture=False,dist=distribution.Normal(0,1),
                  num_classes=None, num_samples=5):

        self.loss_fn = loss_fn

        self.__init_train_params__(latent_size,dist,num_classes,num_samples,True)


        super().__train_loop__(train_loader,gen_optimizer,disc_optimizer,num_epochs, disc_steps,gen_lr_scheduler,disc_lr_scheduler, model_dir, save_model_interval, save_outputs_interval,display_outputs,notebook_mode,batch_log,save_logs,display_metrics,save_metrics,visdom_log,tensorboard_log,save_architecture)


    def __update_discriminator_loss__(self,x,gen_images,real_preds,gen_preds):

        batch_size = x.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        real_labels = Variable(real_labels.cuda() if self.cuda else real_labels)
        fake_labels = Variable(fake_labels.cuda() if self.cuda else fake_labels)

        loss = (self.loss_fn(real_preds - torch.mean(gen_preds),real_labels) + self.loss_fn(gen_preds - torch.mean(real_preds),fake_labels))/2

        return loss

    def __update_generator_loss__(self,x,gen_images,real_preds,gen_preds):
        batch_size = x.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        real_labels = Variable(real_labels.cuda() if self.cuda else real_labels)
        fake_labels = Variable(fake_labels.cuda() if self.cuda else fake_labels)

        loss = (self.loss_fn(real_preds - torch.mean(gen_preds), fake_labels) + self.loss_fn(gen_preds - torch.mean(real_preds),real_labels))/2

        return loss


class HingeGanLearner(StandardBaseGanLearner):
    def __init__(self, gen_model, disc_model, use_cuda_if_available=True):
        super(HingeGanLearner, self).__init__(gen_model, disc_model, use_cuda_if_available)

    def train(self,train_loader, gen_optimizer,disc_optimizer,latent_size,num_epochs=10, disc_steps=1,gen_lr_scheduler=None,disc_lr_scheduler=None, model_dir=os.getcwd(), save_model_interval=1, save_outputs_interval=100,display_outputs=True,notebook_mode=False,batch_log=True,save_logs=None,display_metrics=True,save_metrics=True,visdom_log=None,tensorboard_log=None,save_architecture=False,dist=distribution.Normal(0,1),
                  num_classes=None, num_samples=5):

        self.__init_train_params__(latent_size,dist,num_classes,num_samples,False)


        super().__train_loop__(train_loader,gen_optimizer,disc_optimizer,num_epochs, disc_steps,gen_lr_scheduler,disc_lr_scheduler, model_dir, save_model_interval, save_outputs_interval,display_outputs,notebook_mode,batch_log,save_logs,display_metrics,save_metrics,visdom_log,tensorboard_log,save_architecture)


    def __update_discriminator_loss__(self,x,gen_images,real_preds,gen_preds):

        return torch.mean(F.relu(1-real_preds)) + torch.mean(F.relu(1+gen_preds))

    def __update_generator_loss__(self,x,gen_images,real_preds,gen_preds):

        return -torch.mean(gen_preds)


class RHingeGanLearner(StandardBaseGanLearner):
    def __init__(self, gen_model, disc_model, use_cuda_if_available=True):
        super(RHingeGanLearner, self).__init__(gen_model, disc_model, use_cuda_if_available)

    def train(self,train_loader, gen_optimizer,disc_optimizer,latent_size,num_epochs=10, disc_steps=1,gen_lr_scheduler=None,disc_lr_scheduler=None, model_dir=os.getcwd(), save_model_interval=1, save_outputs_interval=100,display_outputs=True,notebook_mode=False,batch_log=True,save_logs=None,display_metrics=True,save_metrics=True,visdom_log=None,tensorboard_log=None,save_architecture=False,dist=distribution.Normal(0,1),
                  num_classes=None, num_samples=5):

        self.__init_train_params__(latent_size,dist,num_classes,num_samples,True)


        super().__train_loop__(train_loader,gen_optimizer,disc_optimizer,num_epochs, disc_steps,gen_lr_scheduler,disc_lr_scheduler, model_dir, save_model_interval, save_outputs_interval,display_outputs,notebook_mode,batch_log,save_logs,display_metrics,save_metrics,visdom_log,tensorboard_log,save_architecture)


    def __update_discriminator_loss__(self,x,gen_images,real_preds,gen_preds):

        return torch.mean(F.relu(1-real_preds)) + torch.mean(F.relu(1+gen_preds))

    def __update_generator_loss__(self,x,gen_images,real_preds,gen_preds):

        return torch.mean(F.relu(1+real_preds)) + torch.mean(F.relu(1-gen_preds))



class RAvgHingeGanLearner(StandardBaseGanLearner):
    def __init__(self, gen_model, disc_model, use_cuda_if_available=True):
        super(RAvgHingeGanLearner, self).__init__(gen_model, disc_model, use_cuda_if_available)

    def train(self,train_loader, gen_optimizer,disc_optimizer,latent_size,num_epochs=10, disc_steps=1,gen_lr_scheduler=None,disc_lr_scheduler=None, model_dir=os.getcwd(), save_model_interval=1, save_outputs_interval=100,display_outputs=True,notebook_mode=False,batch_log=True,save_logs=None,display_metrics=True,save_metrics=True,visdom_log=None,tensorboard_log=None,save_architecture=False,dist=distribution.Normal(0,1),
                  num_classes=None, num_samples=5):

        self.__init_train_params__(latent_size,dist,num_classes,num_samples,True)


        super().__train_loop__(train_loader,gen_optimizer,disc_optimizer,num_epochs, disc_steps,gen_lr_scheduler,disc_lr_scheduler, model_dir, save_model_interval, save_outputs_interval,display_outputs,notebook_mode,batch_log,save_logs,display_metrics,save_metrics,visdom_log,tensorboard_log,save_architecture)


    def __update_discriminator_loss__(self,x,gen_images,real_preds,gen_preds):

        return (torch.mean(F.relu(1-(real_preds - torch.mean(gen_preds)))) + torch.mean(F.relu(1+(gen_preds - torch.mean(real_preds)))))/2

    def __update_generator_loss__(self,x,gen_images,real_preds,gen_preds):
        return (torch.mean(F.relu(1 + (real_preds - torch.mean(gen_preds)))) + torch.mean(
            F.relu(1 - (gen_preds - torch.mean(real_preds))))) / 2


""" StandardGANModel model suitable for unconditional GANS with an explicit loss function
    Arguments:
    gen_model:  the generator module.
    disc_model:  the discriminator module.
    gen_loss_fn: the loss function to be used for the generator
    disc_loss_fn: the loss function to be used for the discriminator
    smooth_labels: if True, real labels range between 0.7 - 1.2 and fake labels range between 0.0 - 0.3
    use_cuda_if_available: If set to true, training would be done on a gpu if any is available

"""




class WGanLearner(StandardGanCore):
    def __init__(self, gen_model, disc_model, use_cuda_if_available=True):
        super().__init__(gen_model, disc_model, use_cuda_if_available)

    def train(self,train_loader, gen_optimizer,disc_optimizer,latent_size,num_epochs=10, disc_steps=5,gen_lr_scheduler=None,disc_lr_scheduler=None, model_dir=os.getcwd(), save_model_interval=1, save_outputs_interval=100,display_outputs=True,notebook_mode=False,batch_log=True,save_logs=None,display_metrics=True,save_metrics=True,visdom_log=None,tensorboard_log=None,save_architecture=False, dist=distribution.Normal(0,1),
                 real_labels=1, num_classes=None, num_samples=5,lambda_ = 0.25):

        self.lambda_ = lambda_
        self.latent_size = latent_size
        self.dist = dist
        self.fixed_source = None

        self.conditional = (num_classes is not None)
        self.classes = num_classes
        self.num_samples = num_samples

        super().__train_loop__(train_loader,gen_optimizer,disc_optimizer,num_epochs, disc_steps,gen_lr_scheduler,disc_lr_scheduler, model_dir, save_model_interval, save_outputs_interval,display_outputs,notebook_mode,batch_log,save_logs,display_metrics,save_metrics,visdom_log,tensorboard_log,save_architecture)

    def __disc_train_func__(self, data):

        super().__disc_train_func__(data)

        self.disc_optimizer.zero_grad()

        if isinstance(data, list) or isinstance(data, tuple):
            x = data[0]
            if self.conditional:
                class_labels = data[1]
        else:
            x = data

        batch_size = x.size(0)
        if isinstance(self.latent_size, int):
            source_size = list([self.latent_size])
        else:
            source_size = list(self.latent_size)
        source_size = [batch_size] + source_size

        source = self.dist.sample(tuple(source_size))

        if self.cuda:
            x = x.cuda()
            source = source.cuda()

            if self.conditional:
                class_labels = class_labels.cuda()

        x = Variable(x)
        source = Variable(source)
        if self.conditional:
            class_labels = Variable(class_labels.unsqueeze(1))

        if self.conditional:
            outputs = self.disc_model(x, class_labels)
        else:
            outputs = self.disc_model(x)

        if self.conditional:
            random_labels = torch.from_numpy(np.random.randint(0, self.classes, size=(batch_size, 1))).long()
            if self.cuda:
                random_labels = random_labels.cuda()

            random_labels = Variable(random_labels)

            generated = self.gen_model(source, random_labels)
            gen_outputs = self.disc_model(generated.detach(), random_labels)

        else:
            generated = self.gen_model(source)
            gen_outputs = self.disc_model(generated.detach())

        gen_loss = torch.mean(gen_outputs)

        real_loss = -torch.mean(outputs)

        eps = torch.randn(x.size()).uniform_(0, 1)

        if self.cuda:
            eps = eps.cuda()

        x__ = Variable(eps * x.data + (1.0 - eps) * generated.detach().data, requires_grad=True)

        if self.conditional:
            pred__ = self.disc_model(x__,class_labels)
        else:
            pred__ = self.disc_model(x__)

        grad_outputs = torch.ones(pred__.size())

        if self.cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = grad(outputs=pred__, inputs=x__, grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
                         only_inputs=True)[0]

        gradient_penalty = self.lambda_ * ((gradients.view(gradients.size(0), -1).norm(2, 1) - 1) ** 2).mean()

        loss = gen_loss + real_loss + gradient_penalty
        loss.backward()
        self.disc_optimizer.step()

        self.disc_running_loss.add_(loss.cpu() * batch_size)

    def __gen_train_func__(self, data):

        super().__gen_train_func__(data)

        self.gen_optimizer.zero_grad()

        if isinstance(data, list) or isinstance(data, tuple):
            x = data[0]
            if self.conditional:
                class_labels = data[1].unsqueeze(1)
        else:
            x = data
        batch_size = x.size(0)

        if isinstance(self.latent_size, int):
            source_size = list([self.latent_size])
        else:
            source_size = list(self.latent_size)

        source_size = [batch_size] + source_size

        source = self.dist.sample(tuple(source_size))
        if self.conditional:
            random_labels = torch.from_numpy(np.random.randint(0, self.classes, size=(batch_size, 1))).long()
            if self.cuda:
                random_labels = random_labels.cuda()

            random_labels = Variable(random_labels)

        if self.cuda:
            source = source.cuda()

            x = x.cuda()
            if self.conditional:
                class_labels = class_labels.cuda()

        source = Variable(source)
        x = Variable(x)
        if self.conditional:
            class_labels = Variable(class_labels)

        if self.conditional:
            fake_images = self.gen_model(source, random_labels)
            outputs = self.disc_model(fake_images, random_labels)
            real_outputs = self.disc_model(x, class_labels)
        else:
            fake_images = self.gen_model(source)
            outputs = self.disc_model(fake_images)
            real_outputs = self.disc_model(x)

        loss = -torch.mean(outputs)
        loss.backward()

        self.gen_optimizer.step()

        self.gen_running_loss.add_(loss.cpu() * batch_size)


