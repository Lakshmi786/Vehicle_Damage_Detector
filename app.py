
import pandas as pd

from torchvision import transforms
from detecto import core
from detecto import utils
from detecto.visualize import show_labeled_image
from detecto.core import Model
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.pyplot as plt
import gradio as gr
import os
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from pathlib import Path

transform_img = transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize(400),
                                   transforms.RandomHorizontalFlip(0.5),
                                   transforms.ToTensor(),
                                   utils.normalize_transform(),])


labels =  ['damage','BG']
ROOT_DIR = os.getcwd()
path = ROOT_DIR+'/Trained_Model.pth'
print('path',path)
model = Model.load(path, labels) # CHange this while uploading



def prediction_defect(input_image,model = model):
    '''Function takes input of the damaged vehicle
    and provides the damaged area of the vehicle
    '''
    image = utils.read_image(input_image)
    new_image = transform_img(image)
    labels, boxes, scores = model.predict(image)
    top = len(scores[scores > .5])

    return plot_bboxes( input_image, bboxes= boxes[:top],
  xywh=False, labels=labels[:top])



def plot_bboxes(
  image_file: str,
  bboxes: List[List[float]],
  xywh: bool = True,
  labels: Optional[List[str]] = None
) -> None:
    """
    Args:
      image_file: str specifying the image file path
      bboxes: list of bounding box annotations for all the detections
      xywh: bool, if True, the bounding box annotations are specified as
        [xmin, ymin, width, height]. If False the annotations are specified as
        [xmin, ymin, xmax, ymax]. If you are unsure what the mode is try both
        and check the saved image to see which setting gives the
        correct visualization.

    """
    fig = plt.figure()

    # add axes to the image
    ax = fig.add_axes([0, 0, 1, 1])

    image_folder = Path(image_file).parent

    # read and plot the image
    image = plt.imread(image_file)
    plt.imshow(image)

    # Iterate over all the bounding boxes
    for i, bbox in enumerate(bboxes):
        if xywh:
          xmin, ymin, w, h = bbox
        else:
          xmin, ymin, xmax, ymax = bbox
          w = xmax - xmin
          h = ymax - ymin

        # add bounding boxes to the image
        box = patches.Rectangle(
            (xmin, ymin), w, h, edgecolor="red", facecolor="none"
        )

        ax.add_patch(box)

        if labels is not None:
          rx, ry = box.get_xy()
          cx = rx + box.get_width()/2.0
          cy = ry + box.get_height()/8.0
          l = ax.annotate(
            labels[i],
            (cx, cy),
            fontsize=8,
            fontweight="bold",
            color="white",
            ha='center',
            va='center'
          )
          l.set_bbox(
            dict(facecolor='red', alpha=0.5, edgecolor='red')
          )

    plt.axis('off')
    outfile = os.path.join(image_folder, "image_bbox.jpg")
    fig.savefig(outfile)

    print("Saved image with detections to %s" % outfile)
    return outfile


gr.Interface(fn=prediction_defect, 
            
           inputs = [ gr.inputs.Image(type="filepath", label="Please Upload the Defect Image") ],
             outputs= [gr.outputs.Image(type="pil")],
             examples=[]).launch(debug= True)









