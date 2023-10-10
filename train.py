# Imports here
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

#----------------------------------



def arg_parse():
    parser = argparse.ArgumentParser(description="parser for train.py")
    
    parser.add_argument("--data_dir", type=str, default="flowers", help="Directory with all data")
    parser.add_argument("--save_dir", type=str, default="checkpoint_3.pth", help="Directory to checkpoints")
    parser.add_argument("--arch", type=str, default="vgg16", choices=["vgg16", "densenet121"], help="Pretrained model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=500, help="No. of hidden units")
    parser.add_argument("--epochs", type=int, default=6, help="No. of epochs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to train the model")
    
    return parser.parse_args()
#------------------------------------------
def main():
    args = arg_parse()

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms =transforms.Compose([transforms.RandomRotation(255),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)


    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

        # TODO: Build and train your network
    learning_rate = 0.001
    device = torch.device(args.device)
    model = getattr(models, args.arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    input_units = 25088 if args.arch == "vgg16" else 1024
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_units, args.hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.1)),
        ('fc2', nn.Linear(args.hidden_units, len(cat_to_name))),
        ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            steps += 1
            #set gradients to zero
            optimizer.zero_grad()
            #forward pass
            logps = model.forward(images)
            #calculate loss
            loss = criterion(logps, labels)
            #backward pass
            loss.backward()
            #update weights
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                accuracy = 0
                validation_loss = 0
                #Evaluate model
                model.eval()
                #turn off gradients
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        loss = criterion(logps, labels)
                        validation_loss += loss.item()
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                        f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': input_units,
                  'output_size': len(cat_to_name),
                  'pretrained_model': getattr(models, args.arch)(pretrained=True),
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer,
                  'learning_rate': learning_rate,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx
                  }
    # TODO: Save the checkpoint 
    torch.save(checkpoint, args.save_dir)

print("Training is complete")
#------------------------------------------
if __name__ == "__main__":
    main()