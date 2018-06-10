import torchfusion as tf
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.squeezenet import squeezenet1_1
from PIL import Image

INFER_FOLDER  = r"C:\Users\Moses\Documents\Moses\W7\AI\ImageAI\ImageAI 1.0.2 Repo\images"
MODEL_PATH = r"C:\Users\Moses\Documents\Prime Project\AI\PyTorch\tests\squeezenet.pth"

transformations = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

infer_set = tf.ImagesFromPaths([INFER_FOLDER],recursive=False,transformations=transformations)
infer_loader = DataLoader(infer_set,batch_size=2)

net = squeezenet1_1()

model = tf.StandardModel(net)
model.load_model(MODEL_PATH)

def predict_loader(data_loader):
    predictions = model.predict(data_loader,apply_softmax=True)
    print(len(predictions))

    for pred in predictions:
        class_index = torch.argmax(pred)
        class_name = tf.decode_imagenet(class_index)
        confidence = torch.max(pred)
        print("Prediction: {} , Accuracy: {} ".format(class_name, confidence))

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transformations(img)
    img = img.unsqueeze(0)

    pred = model.predict(img,apply_softmax=True)

    class_index = torch.argmax(pred)
    class_name = tf.decode_imagenet(class_index)
    confidence = torch.max(pred)

    print("Prediction: {} , Accuracy: {} ".format(class_name, confidence))


if __name__ == "__main__":

    predict_loader(infer_loader)
    predict_image(r"C:\Users\Moses\Documents\Moses\W7\AI\ImageAI\ImageAI 1.0.2 Repo\sample.jpg")

