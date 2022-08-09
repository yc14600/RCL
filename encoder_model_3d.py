import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.training import Replay
from models.resnet18_encoder import ResNetEncoder
from benchmarks.SplitCifar import SplitCIFAR10

# utility functions to create plugin metrics
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics
import os




# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNetEncoder(nclasses=10)
# CL Benchmark Creation
perm_mnist = SplitCIFAR10(n_experiences=1, dataset_root='images')

train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9)
criterion = CrossEntropyLoss()


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


# Continual learning strategy

cl_strategy = Replay(
    model, optimizer, criterion, mem_size=1500, train_epochs=5, train_mb_size=50, eval_mb_size=50,
    device=device, evaluator=eval_plugin)


# train and test loop over the stream of experiences
results = []
for train_exp in train_stream:
    cl_strategy.train(train_exp)
    results.append(cl_strategy.eval(test_stream))

torch.save(model.state_dict(), os.getcwd()+'/encoder_model.pth')