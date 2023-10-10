# Imports here

import json

import matplotlib.pyplot as plt

import argparse
import numpy as np
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time
#--------------------------------------------------------------
def arg_parse():
    parser = argparse.ArgumentParser(description="parser for predict.py")
    
    parser.add_argument("--img_path", type=str, help="Testing image path")
    parser.add_argument("--checkpoint", type=str,default='checkpoint_2.pth', help="Saved trained model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Top K=")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Json file to change folder labels to names")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to train on")
    
    return parser.parse_args()

#--------------------------------------------------------------

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = checkpoint['pretrained_model']

    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    model.learning_rate = checkpoint['learning_rate']
    return model

#--------------------------------------------------------------

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # Get image original sizes
    test_transforms =transforms.Compose([transforms.RandomRotation(255),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    img = test_transforms(image).numpy()
    return torch.Tensor(img)

#--------------------------------------------------------------

def predict(device, image_path, model, topk=5):
    ''' Predict the top K classes of an image using a trained model.
    
    Arguments:
        image_path: Path to the image file.
        model: Trained PyTorch model.
        topk: Number of top predicted classes to return.
    
    Returns:
        top_probs: List of top K predicted probabilities.
        top_classes: List of top K predicted class labels.
    '''
    with Image.open(image_path) as image:
        processed_image = process_image(image)

    model.eval()
    model.to(device)
    
    processed_image = processed_image.to(device)  # Move the processed image tensor to the GPU
    processed_image = processed_image.float()
    processed_image.unsqueeze_(0)  # Add a batch dimension

    # Use the model to make predictions
    with torch.no_grad():
        logps = model(processed_image)
        # Calculate probability
        ps = torch.exp(logps).cpu()
        top_ps, top_class = ps.topk(topk, dim=1)
        
        top_ps = np.array(top_ps[0])
        top_class = np.array(top_class[0])
    
    model.train()
    
    # Convert indices to actual category names
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [index_to_class[idx] for idx in top_class]
    
    return top_ps, top_class

#--------------------------------------------------------------
def main():
    args = arg_parse()
    
    model = load_checkpoint(args.checkpoint)
    
    top_ps, top_class = predict(args.device,args.img_path, model, args.top_k)
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    top_flowers = [cat_to_name[cat] for cat in top_class]
    
    print(f"Top {args.top_k} flowers: {top_flowers}")
    print(f"Top {args.top_k} probabilities: {top_ps}")