
#TorchFusion

Training Neural Networks with Torchfusion.

Torchfusion provides an out-of the box StandardModel class that allows you to train almost any network for classification or regression
It also provides a BaseModel class that makes it easy for you to write trainers with custom training logic.
In the first part of this tutorial, you would be introduced to how to train neural networks with the StandardModel.


<h1> Training MNIST </h1>
MNIST is considered the "Hello World" of Deep Learning, rightly so because it is the simplest image recognition task.
Just in case you are new to Deep Learning, MNIST is a dataset of handwritten digits. Every image is labelled according to the digit
it represents.
Below are some examples from the training set
///

The dataset is divided into 60 000 training images and 10 000 test images.
For more info about MNIST, visit (http://yann.lecun.com/exdb/mnist)

Before you proceed further, ensure you have installed TorchFusion,Pytorch and TorchVision following instructions in the [Intro](Intro)


Fireup your IDE, (I always love PyCharm, Submlime Text is cool too)

<h2>Step1: Import required packages</h2>
<pre>
import torchfusion as tf
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
from torch.optim import Adam
import torch.cuda as cuda

</pre>

Never mind the many imports, only the first is from torchfusion, the others are needed for pytorch and torchvision,
It's perfectly fine if you just copy and paste the imports above, you would understand them soon.
<pre>
Step2: Define Data augmentations
transformations = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

</pre>

Using the transforms class we imported from torchvision, we have to compose a number of data augmentations that first resizes the images to 28 x 28,
Next, PyTorch only understands Tensors, so we have to convert the image to a tensor, finally, we normalize the tensor to range between -1 and 1



Step3: Load the images
<pre>

batch_size = 64

train_set = MNIST(root="./data",transform=transformations,download=True)
test_set = MNIST(root="./data",train=False,transform=transformations,download=True)

train_loader = DataLoader(train_set,shuffle=True,batch_size=batch_size,num_workers=4)
test_loader = DataLoader(test_set,shuffle=False,batch_size=batch_size,num_workers=4)
</pre>

Step4: Define the network
<pre>
net = nn.Sequential(
            tf.Flatten(),
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
)

</pre>

The above is a simple fully connected network. The first layer would flatten the 28 x 28 images into 784. 
Next, are three hidden Linear layers, each of which is followed by ReLU. Finally, the last fully connected layer outputs scores for every single single class.
In this case, there are ten classes 0 - 9.

Step5: Move the network to the GPU if available
<pre>
if cuda.is_available():
    net.cuda()
</pre>

The above step is very important especially when you are using much larger networks. The only reason why "Deep Learning" is possible is the 
availability of highly optimized GPUs from NVIDIA. Neural networks often always involves large matrix multplication and dot products, these are parallel 
operations and GPUs are specially optimized for parallel operations.

Step6: Setup the optimizer and loss function
<pre>
optimizer = Adam(net.parameters(),lr=0.001)
loss_fn = nn.CrossEntropyLoss()
</pre>

Neural networks works by multiplying a matrix of numbers called weights with the inputs.
Every layer above except the Flatten layer has its own weights.
The singular purpose of training neural networks is to find the best values for these weights.

To do so, an optimization algorithm named "Stochastic Gradient Descent is used" (SGD)
Here we use a variant of SGD, named Adam. It is a very stable variant with excellent convergence properties.

Lr above refers to the learning rate, this is the speed at which the weights are adjusted, it should never be too high nor too low.
For Adam, 0.001 is a great starting value.

The loss function, tells the network how bad or good it is performing, this helps the optimizer to adjust the weights in the right direction.
Here we use the CrossEntropyLoss, which is particularly good for classification tasks.

Step7: Define metrics
<pre>
#Top 1 Train accuracy
train_metrics = tf.Accuracy(topK=1)

#Top 1 and Top 2 train accuracy
test_metrics_top1 = tf.Accuracy(name="Top 1 Acc ",topK=1)
test_metrics_top2 = tf.Accuracy(name="Top 2 Acc ",topK=2)

</pre>

Metrics are not used to adjust weights, but they provide an easy to interprete indicator of how well our network is performing.
The Accuracy metric above tells us how often the prediction is correct.

Step8: Create the trainer 
<pre>

model = tf.StandardModel(net)
print(model.summary((1,28,28)))

</pre>
Here we use the StandardModel provided by TorchFusion. Notice that we pass the network into it.
The second line prints a comprehensive summary of the network

Step9:
<pre>
if __name__ == "__main__":
    
    model.train(train_loader, loss_fn, optimizer, [train_metrics], test_loader,
                [test_metrics_top1, test_metrics_top2], num_epochs=20,
                model_dir="mnist_mlp_saved_models", notebook_mode=False, display_metrics=False, batch_log=True,
                save_logs="logs.txt")

</pre>
Here is the most important part, where we fuse together everything we have defined so far.
The num_epochs defines how many times the training would go over the entire training set.

N.B: Always run this part in the "__main__" loop


<h1>Extra Time </h1>
The above are the basics, this part covers a few extra topics including adjusting learning rates and customizing the training

<h3>Adjusting Learning Rates</h3>
<pre>
def lr_schedule(epoch):

    lr = 0.001

    if epoch > 90:
        lr /= 1000
    elif epoch > 60:
        lr /= 100
    elif epoch > 30:
        lr /= 10
    print("LR: ",lr)
    return lr

</pre>

The function above specifies the learning rate based on the current epoch, in the code below, we apply it to our training

<pre>
 model.train(train_loader, loss_fn, optimizer, [train_metrics], test_loader,
                [test_metrics_top1, test_metrics_top2], num_epochs=20,
                model_dir="mnist_mlp_saved_models",notebook_mode=False,display_metrics=False,lr_schedule=lr_schedule,batch_log=True,save_logs="logs.txt")
</pre>

<h3>Metrics</h3>
Presently, torchfusion only includes topK metrics and MSE
The topK metrics provides an excellent metric for knowing how often the correct class appears in the topK predictions.

Doing this is simple in torchfusion, just specify the topK value and optionally a name

<pre>
top1 = tf.Accuracy(name="Top 1 Acc ",topK=1)
top2 = tf.Accuracy(name="Top 2 Acc ",topK=2)
top5 = tf.Accuracy(name="Top 5 Acc ",topK=5)
</pre>

<h3>Logging</h3>

you can easily enable and disable various logging mechanisms using a few arguments in the train function
<pre>
display_metrics: When true, plots of all the metrics and the loss function are shown at the end of every single epoch
save_metrics: When true, plots of all the metrics and the loss function are shown saved to the model_dir folder
batch_log: Logs the train metrics and running loss after every single batch
notebook_mode: If batch_log is true, the logging is optimized for notebook environments, note that when running on Google Colab, this raises error due to 
unavailablity of Ipywidgets on Google Colab. It is recommended to set this when working in Colab, (Until Google fixes this)
save_logs: This specifies a path in which to save training logs, this is particlularly useful when running experiments whereby visual supervision is not possible.
model_dir: Directory in which to save models and metrics
save_models: This should be "all" or "best", the first mode saves all models in the all_models subfolder of the model_dir and it also saves the best models
in the best_models subfolder. If "best", only the best models are saved
</pre>
<h1>Creating a Custom Trainer </h1>

At the heart of torchfusion is the BaseModel class that provides a number of predefined functionalities that will serve as the foundation for most of the
trainers you would use.

You can easily create your own custom trainers that extend the BaseModel, here we shall recreate the StandardModel as a demonstration
There are three functions you need to override when creating your own trainers.
They are __train_func__ , __eval_function__  and __predict_function__

Step1: Override the BaseModel
<pre>
    def __init__(self,model,use_cuda_if_available=True):
        super(StandardModel,self).__init__(model,use_cuda_if_available)

</pre>
The StandardModel takes in a single network(model)

Step2: Override the __train_func__
<pre>

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

</pre>
As you can see above, the __train_func__ does does not handle anything related to iterating over epochs and fetching batches,
these are done by the BaseModel, our train function only needs to define the logic for training, update the running_loss and the metrics.


Step2: Override the __eval_func__
<pre>

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
</pre>
Again this function only states the logic for evaluating and update the metrics, loading the data and printing logs are entirely up to the BaseModel


Step3: Override the __predict_func__

<pre>

    def __predict_func__(self,inputs):

        if self.cuda:
            inputs = inputs.cuda()

        inputs = Variable(inputs)
        output = self.model(inputs)

        return output

</pre>
The base model handles user inputs gracefully, both dataloaders and tensors.

As you can see above, creating custom trainers is very easy, you only need to define the logic, all common, repeative operations such as loading the data,
handling callbacks, printing metrics, logging etc. This helps you focus purely on the logic as a researcher.
