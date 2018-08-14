import torch
import random
from torch.autograd import Variable


class ImagePool():
    def __init__(self,pool_size):

        self.pool_size = 0
        if self.pool_size > 0:
            self.image_array = []
            self.num_imgs = 0


    def query(self,input_images):

        if isinstance(input_images,Variable):
            input_images = input_images.data

        if self.pool_size == 0:
            return input_images

        ret_images = []

        for image in input_images:
            image = image.unsqueeze(0)

            if self.num_imgs < self.pool_size:
                self.image_array.append(image)
                self.num_imgs += 1
                ret_images.append(image)

            else:
                prob = random.uniform(0,1)

                if prob > 0.5:
                    random_image_index = random.randint(0,self.pool_size - 1)
                    ret_images.append(self.image_array[random_image_index])
                    self.image_array[random_image_index] = image
                else:
                    ret_images.append(image)

            return torch.cat(ret_images,0)

