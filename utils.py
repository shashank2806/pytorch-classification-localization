import torch
import torchvision
from PIL import Image
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchvision import datasets, transforms, models

SAVE_PATH = 'model/model.pth'
class_names = ['nospill', 'spill']

def load_ckpt(ckpt_path):
    # loads saved model from a checkpoint file
    ckpt = torch.load(ckpt_path)
    
    model = models.densenet121(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False

    # Put the classifier on the pretrained network
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.load_state_dict(ckpt, strict=False)

    return model


# load model
model = load_ckpt(SAVE_PATH)

test_transforms = transforms.Compose([transforms.Resize(224),
                                    transforms.RandomCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])


def process_image(image):
    ''' Scales, crops, and normalizes a CV2 image for a PyTorch model,
        returns an Numpy array
    '''
    return test_transforms(image)


def predict(image_path, model):
    # Predict the class (or classes) of an image using a trained deep learning model.
    model.eval()
    img_pros = process_image(image_path)
    img_pros = img_pros.view(1,3,224,224)
    output = model(img_pros)
    # probability of each classs
    ps = torch.exp(output)
    top_p, top_class = ps.topk(1, dim=1)
    return ps, class_names[top_class]






