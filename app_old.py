import pandas as pd

from torchvision import transforms
from detecto import core
from detecto import utils
from detecto.visualize import show_labeled_image
from detecto.core import Model
import matplotlib.pyplot as plt
import matplotlib.image as img


transform_img = transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize(400),
                                   transforms.RandomHorizontalFlip(0.5),
                                   transforms.ToTensor(),
                                   utils.normalize_transform(),])


labels =  ['damage','BG']
model = Model.load('/content/drive/MyDrive/Trained_Model.pth', labels)

def prediction_defect(new_image,model = model):
    '''Function takes input of the damaged vehicle
    and provides the damaged area of the vehicle
    '''
    image = utils.read_image(new_image)
    new_image = transform_img(image)
    labels, boxes, scores = model.predict(image)
    top = len(scores[scores > .5])
    show_labeled_image(image, boxes[:top], labels[:top])