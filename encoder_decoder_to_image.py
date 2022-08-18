import torchvision.transforms as transforms    # to normalize, scale etc the dataset
from torch.utils.data import DataLoader        # to load data into batches (for SGD)
from torchvision.utils import make_grid        # Plotting. Makes a grid of tensors
from torchvision.datasets import MNIST         # the classic handwritten digits dataset
import matplotlib.pyplot as plt                # to plot our images
import numpy as np
import torch
import os
from models.generator import MlpVAE as MlpVAE_model

def image_generator(counter, images, testset,path,device='cpu'):
    device = torch.device(device)
    #testset  = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    # Create DataLoader objects. These will give us our batches of training and testing data.
    #batch_size = 100
    #testloader  = DataLoader(testset, batch_size=batch_size, shuffle=True)

    #model = MlpVAE_model(shape=(1, 28, 28), n_classes=10)
    #model.load_state_dict(torch.load(os.getcwd()+'/encoder-decoder_model.pth'))
    #model.eval()


    #test_images = []
    #for images, _ in testloader:
        #test_images.append(images)
        #break

    test_images = images
    reconstructions = testset
    with torch.no_grad():
        print("Original Images")
        fig = plt.figure()
        test_images = test_images.to(device)
        test_images = test_images.cpu()
        test_images = test_images.clamp(0, 1)
        test_images = test_images[:50]
        test_images = make_grid(test_images, 10, 5)
        test_images = test_images.numpy()
        test_images = np.transpose(test_images, (1, 2, 0))
        plt.imshow(test_images)
        plt.savefig(os.path.join(path,'images_previous','image'+str(counter)+'.jpg'))


        #reconstructions = reconstructions.view(reconstructions.size(0), 1, 28, 28)
        reconstructions = reconstructions.cpu()
        reconstructions = reconstructions.clamp(0, 1)
        reconstructions = reconstructions[:50]
        reconstructions = np.transpose(make_grid(reconstructions, 10, 5).numpy(), (1, 2, 0))
        plt.imshow(reconstructions)
        plt.savefig(os.path.join(path,'images_after','image'+str(counter)+'.jpg'))

__all__ = ["image_generator"]