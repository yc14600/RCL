import copy

import torch
from models.resnet18_encoder import ResNetEncoder
from models.resnet_generator import ResnetVAE
from models.resnet_generator import VAE_loss as VAE_LOSS
from benchmarks.SplitCifar import *
from avalanche.logging import TensorboardLogger, TextLogger
from training.VAEtraining import VAEEWCTraining as VAETRAINING
from torch.optim import Adam
from avalanche.training.plugins import (
    EvaluationPlugin,

)
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.training import Replay, EWC
from encoder_decoder_to_image import image_generator
from avalanche.logging import InteractiveLogger
import os
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, StreamConfusionMatrix


# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model
#model = ResnetVAE(z_dim=20)
model = ResnetVAE(z_dim=32)
encoder_model = ResNetEncoder(nclasses=10, z_dim=32)




# CL Benchmark Creation

perm_mnist = SplitCIFAR10(n_experiences=5, dataset_root='images')
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream


# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))
eval_plugin = EvaluationPlugin(
    # accuracy after each training epoch
    # and after each evaluation experience
    accuracy_metrics(epoch=True, experience=True),
    # loss after each training minibatch and each
    # evaluation stream
    loss_metrics(minibatch=True, stream=True),
    # catastrophic forgetting after each evaluation
    # experience
    forgetting_metrics(experience=True, stream=True),
    StreamConfusionMatrix(num_classes=10, save_image=False),
    # add as many metrics as you like
    loggers=[InteractiveLogger(), text_logger, TensorboardLogger()],
    benchmark = perm_mnist
)

optimizer = SGD(encoder_model.parameters(), lr=0.005, momentum=0.9)
criterion = CrossEntropyLoss()
encoder_strategy = EWC(
    encoder_model, optimizer, criterion, train_epochs=5,  train_mb_size=50, eval_mb_size=50,
    device=device, evaluator=eval_plugin, ewc_lambda=1)
# train and test loop over the stream of experiences



# log to text file
text_logger = TextLogger(open('log.txt', 'a'))
eval_plugin = EvaluationPlugin(
    loss_metrics(minibatch=True, stream=True),
    loggers=[InteractiveLogger(), text_logger, TensorboardLogger()],
    benchmark = perm_mnist
)

cl_strategy = VAETRAINING(model, optimizer=Adam(model.parameters(), lr=0.001, weight_decay=1e-5),
                              criterion=VAE_LOSS,
                              train_epochs=10, mem_size=1000, device=device, evaluator=eval_plugin, train_mb_size=100,
                              eval_mb_size=100, ewc_lambda=1)

# Continual learning strategy evaluator=eval_plugin





representations = []
counter = 0
os.mkdir(os.getcwd() + '/results_images')
os.mkdir(os.getcwd() + '/results_images/images_previous')
os.mkdir(os.getcwd() + '/results_images/images_after')
test_stream_2 = copy.deepcopy(test_stream)
for train_exp, test_exp in zip(train_stream, test_stream):
    counter += 1
    train_exp_2 = copy.deepcopy(train_exp)
    print("Begin encoder training "+str(counter))
    encoder_strategy.train(train_exp)
    encoder_strategy.eval(test_stream)
    print("End encoder training "+str(counter))
    for i in encoder_model.features1.parameters():
        i.requires_grad = False
    for i in encoder_model.features2.parameters():
        i.requires_grad = False
    model.encoder.features1 = encoder_model.features1
    model.encoder.features2 = encoder_model.features2
    cl_strategy.model = model
    print("Begin decoder training"+str(counter))
    cl_strategy.train(train_exp)
    print("End decoder training" + str(counter))
    for i in encoder_model.features1.parameters():
        i.requires_grad = True
    for i in encoder_model.features2.parameters():
        i.requires_grad = True
    representations, images, results = cl_strategy.eval(test_stream_2)
    # images.pop()
    # results.pop()
    # before = len(representations) - 5
    # after = len(representations) - 5 + experience_number
    # representations = representations[before:after]
    # images = images[before: after]
    # results = results[before: after]
    images = [x for xs in images for x in xs]
    representations = [x for xs in representations for x in xs]
    results = [x for xs in results for x in xs]
    # experience_number += 1


    channels, size_x, size_y = results[0].size()[1], results[0].size()[2], results[0].size()[3]
    results=torch.stack(results).view(-1, channels, size_x, size_y)
    images =torch.stack(images).view(-1, channels, size_x, size_y)
    indexes = torch.randperm(results.shape[0])
    results, images = results[indexes], images[indexes]
    image_generator(counter, images, results)


torch.save(model.state_dict(), os.getcwd()+'/encoder-decoder_model.pth')