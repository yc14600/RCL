import torch
from models.encoder_model import *
from models.generator import MlpVAE as MlpVAE_model
from models.resnet_generator import ResnetVAE
from models.generator import VAE_loss as VAE_LOSS
from models.resnet_generator import VAE_loss as VAE_LOSS_resnet
from benchmarks.SplitMnist import *
from benchmarks.SplitCifar import *
from avalanche.logging import TensorboardLogger, TextLogger
from avalanche.evaluation.metrics import loss_metrics
from training.VAEtraining import VAETraining as VAETRAINING
from torch.optim import SGD, Adam
from avalanche.training.plugins import (
    EvaluationPlugin,

)
from encoder_decoder_to_image import image_generator
from avalanche.logging import InteractiveLogger
import os

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model
#model = ResnetVAE(z_dim=20)
model = MlpVAE_model(shape=(1, 28, 28), n_classes=10)
encoder_model = encoder_model()
encoder_model.load_state_dict(torch.load(os. getcwd()+'/encoder_model.pth'))
for i in encoder_model.features.parameters():
    i.requires_grad = False
    print(i)
model.encoder.encode = encoder_model.features


# CL Benchmark Creation

perm_mnist = SplitMNIST(n_experiences=5, dataset_root='images')
#perm_mnist = SplitCIFAR10(n_experiences=1)
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream


# Prepare for training & testing
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)



# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))
eval_plugin = EvaluationPlugin(
    loss_metrics(minibatch=True, stream=True),
    loggers=[InteractiveLogger(), text_logger, TensorboardLogger()],
    benchmark = perm_mnist
)


# Continual learning strategy evaluator=eval_plugin



cl_strategy = VAETRAINING(
    model, optimizer, criterion=VAE_LOSS, train_epochs=5, mem_size=1000, device=device, evaluator=eval_plugin, train_mb_size=100, eval_mb_size=100)

# train and test loop over the stream of experiences

metrics = []
counter = 0
os.mkdir(os.getcwd() + '/results_images')
os.mkdir(os.getcwd() + '/results_images/images_previous')
os.mkdir(os.getcwd() + '/results_images/images_after')
for train_exp, test_exp in zip(train_stream, test_stream):
    counter += 1
    cl_strategy.train(train_exp)
    metric, images, results = cl_strategy.eval(test_exp)
    images.pop()
    results.pop()
    channels, size_x, size_y = results[0].size()[1], results[0].size()[2], results[0].size()[3]
    results=torch.stack(results).view(-1, channels, size_x, size_y)
    images =torch.stack(images).view(-1, channels, size_x, size_y)
    indexes = torch.randperm(results.shape[0])
    results, images = results[indexes], images[indexes]
    metrics.append(metric)
    torch.save(model.state_dict(), os.getcwd() + '/encoder-decoder_model.pth')
    image_generator(counter, images, results)
    os.remove(os.getcwd() + '/encoder-decoder_model.pth')

for i in model.encoder.encode.parameters():
    print(i)
torch.save(model.state_dict(), os.getcwd()+'/encoder-decoder_model.pth')