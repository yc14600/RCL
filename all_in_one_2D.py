import copy
import os
import argparse
import numpy as np

import torch
from models.encoder_model import *
from models.generator import MlpVAE as MlpVAE_model
from models.generator import VAE_loss as VAE_LOSS
from benchmarks.SplitMnist import *
from avalanche.logging import TensorboardLogger, TextLogger, CSVLogger
from training.VAEtraining import VAEReplayTraining, VAECWRTraining, VAENaiveTraining, VAEEWCTraining, VAELWFTraining
from training.classificator_training import LWF, Replay, EWC, Naive, CWR
from torch.optim import Adam
from avalanche.training.plugins import (
    EvaluationPlugin,

)

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from encoder_decoder_to_image import image_generator
from avalanche.logging import InteractiveLogger
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, StreamConfusionMatrix
from utils import *


parser = argparse.ArgumentParser()

parser.add_argument('-sd','--seed', default=0, type=int, help='random seed')
parser.add_argument('-cuda','--use_cuda', default=False, type=str2bool, help='use cuda')
parser.add_argument('-rpth','--result_path', default='./results', type=str, help='path to save results')
parser.add_argument('-dpth','--data_path', default='../../dataset', type=str, help='path to dataset')
parser.add_argument('-ep','--epoch', default=10, type=int, help='number of epochs')
parser.add_argument('-stype','--strategy_type', default='replay', type=str, help='continual learning strategy type')
parser.add_argument('-bmk','--benchmark', default='splitmnist', type=str, help='benchmark')
parser.add_argument('-T','--T', default=5, type=int, help='number of tasks')
parser.add_argument('-C','--C', default=10, type=int, help='number of classes')
parser.add_argument('-mtype','--model_type', default='MLP', type=str, help='model type')
parser.add_argument('-lr','--learning_rate', default=0.001, type=float, help='learning rate for training localization MLPs')
parser.add_argument('-bsz','--batch_size', default=128, type=int, help='batch size for training MLPs')
parser.add_argument('-zd','--z_dim', default=32, type=int, help='number of latent dimentions')
parser.add_argument('-ms','--mem_size', default=100, type=int, help='memory size')
parser.add_argument('-dvc','--device', default='cpu', type=str, help='device')


args = parser.parse_args()
print(args)

seed = args.seed
print('seed',seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Config
device = torch.device(args.device)

#model
#model = ResnetVAE(z_dim=20)
if args.model_type == 'MLP':
    model = MlpVAE_model(shape=(1, 28, 28), n_classes=args.C)


encoder_model = encoder_model()


rpath,dpath = config_result_path(args)

# CL Benchmark Creation
if args.benchmark == 'splitmnist':
    benchmark = SplitMNIST(n_experiences=args.T, dataset_root=args.data_path)
#benchmark = SplitCIFAR10(n_experiences=1)
train_stream = benchmark.train_stream
test_stream = benchmark.test_stream


# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
#text_logger = TextLogger(open(os.path.join(rpath,'log.txt'), 'a'))
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
    StreamConfusionMatrix(num_classes=args.C, save_image=False),
    # add as many metrics as you like
    loggers=[InteractiveLogger(), CSVLogger(log_folder=rpath)],
    benchmark = benchmark
)

optimizer = SGD(encoder_model.parameters(), lr=args.learning_rate, momentum=0.9)
criterion = CrossEntropyLoss()
if args.strategy_type == 'replay':
    encoder_strategy = Replay(
        encoder_model, optimizer, criterion, train_epochs=1, mem_size=args.mem_size,  train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
        device=device, evaluator=eval_plugin)
# train and test loop over the stream of experiences



# log to text file
#text_logger = TextLogger(open(os.path.join(dpath,'log.txt'), 'a'))
eval_plugin = EvaluationPlugin(
    loss_metrics(minibatch=True, stream=True,experience=True),
    loggers=[InteractiveLogger(), CSVLogger(log_folder=dpath)],
    benchmark = benchmark
)
if args.strategy_type == 'replay':
    decoder_strategy = VAEReplayTraining(model, optimizer=Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5),
                          criterion=VAE_LOSS,
                          train_epochs=args.epoch, device=device, mem_size=args.mem_size, evaluator=eval_plugin, train_mb_size=args.batch_size,
                          eval_mb_size=args.batch_size)
# Continual learning strategy evaluator=eval_plugin





metrics = []
counter = 0
experience_number = 1


test_stream_2 = copy.deepcopy(test_stream)
for train_exp, test_exp in zip(train_stream, test_stream):
    counter += 1
    train_exp_2 = copy.deepcopy(train_exp)
    print("Begin encoder training "+str(counter))
    encoder_strategy.train(train_exp)
    encoder_strategy.eval(test_stream)
    print("End encoder training "+str(counter))
    for i in encoder_model.features.parameters():
        print(i)
        i.requires_grad = False

    model.encoder.encode = encoder_model.features
    decoder_strategy.model = model

    print("Begin decoder training"+str(counter))
    decoder_strategy.train(train_exp)
    print("End decoder training" + str(counter))

    for i in encoder_model.features.parameters():
        i.requires_grad = True

    representations, images, results = decoder_strategy.eval(test_stream_2)

    before = len(representations)-5
    after = len(representations)-5+ experience_number
    representations = representations[before:after]
    images = images[before: after]
    results = results[before: after]
    images = [x for xs in images for x in xs]
    representations = [x for xs in representations for x in xs]
    results = [x for xs in results for x in xs]
    experience_number += 1

    channels, size_x, size_y = results[0].size()[1], results[0].size()[2], results[0].size()[3]
    results=torch.stack(results).view(-1, channels, size_x, size_y)
    images =torch.stack(images).view(-1, channels, size_x, size_y)
    indexes = torch.randperm(results.shape[0])
    results, images = results[indexes], images[indexes]
    image_generator(counter, images, results,path=dpath,device=args.device)


#torch.save(model.state_dict(),os.path.join(rpath,'encoder-decoder_model.pth'))