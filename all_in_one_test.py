from base64 import encode
import copy
import argparse
import numpy as np

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim import SGD


from avalanche.logging import InteractiveLogger, TextLogger, CSVLogger
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, StreamConfusionMatrix
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins import ReplayPlugin, GDumbPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer
#from benchmarks.SplitMnist import *
#from benchmarks.SplitCifar import *
from avalanche.benchmarks.classic import *
from training.VAEtraining import VAEReplayTraining, VAENaiveTraining
from training.classifier_training import TrainStrategy
from models.encoder_model import *
from models.generator import MlpAE 
from models.generator import AE_LOSS
from models.resnet18_encoder import ResNetEncoder
from models.resnet_generator import ResnetAE
from encoder_decoder_to_image import image_generator
from utils import *

benchmarks = {'splitmnist':SplitMNIST,'splitcifar10':SplitCIFAR10, 'splitcifar110':SplitCIFAR110, \
    'splitcifar100':SplitCIFAR100,'splittinyimagenet':SplitTinyImageNet}

parser = argparse.ArgumentParser()

parser.add_argument('-sd','--seed', default=0, type=int, help='random seed')
parser.add_argument('-rpth','--result_path', default='./results', type=str, help='path to save results')
parser.add_argument('-dpth','--data_path', default='../../dataset', type=str, help='path to dataset')
parser.add_argument('-ep','--epoch', default=10, type=int, help='number of epochs')
parser.add_argument('-stype','--strategy_type', default='naive', type=str, help='continual learning strategy type')
parser.add_argument('-dstype','--decoder_strategy', default='replay', type=str, help='continual learning strategy type')
parser.add_argument('-bmk','--benchmark', default='splitmnist', type=str, help='benchmark')
parser.add_argument('-T','--T', default=5, type=int, help='number of tasks')
parser.add_argument('-C','--C', default=10, type=int, help='number of classes')
parser.add_argument('-mtype','--model_type', default='MLP', type=str, help='model type')
parser.add_argument('-lr','--learning_rate', default=0.001, type=float, help='learning rate for training localization MLPs')
parser.add_argument('-bsz','--batch_size', default=128, type=int, help='batch size for training MLPs')
parser.add_argument('-zd','--z_dim', default=32, type=int, help='number of latent dimentions')
parser.add_argument('-ms','--mem_size', default=100, type=int, help='memory size')
parser.add_argument('-dvc','--device', default='cpu', type=str, help='device')
parser.add_argument('-rpl','--replay_strategy', default='replay', type=str, help='replay strategy')



args = parser.parse_args()
print(args)

seed = args.seed
print('seed',seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Config
device = torch.device(args.device)
rpath,dpath = config_result_path(args)

#model

if 'mnist' in args.benchmark:
    args.model_type = 'MLP'
    in_dim = 784
    shape = (1, 28, 28)
    enc_model = encoder_model(input_size = in_dim, shape=shape, latent_dim=args.z_dim)
    model = MlpAE(shape=shape, n_classes=args.C,latent_dim=args.z_dim,encoder=enc_model)
    
elif 'cifar' in args.benchmark: 
    args.model_type == 'resnet'
    enc_model = ResNetEncoder(nclasses=args.C, z_dim=args.z_dim)
    model = ResnetAE(z_dim=args.z_dim,encoder=enc_model)
    
    
# CL Benchmark Creation
if args.benchmark!='splitcifar110':
    benchmark = benchmarks[args.benchmark](n_experiences=args.T, dataset_root=args.data_path)
else:
    benchmark = benchmarks[args.benchmark](n_experiences=args.T, dataset_root_cifar10=args.data_path,dataset_root_cifar100=args.data_path)

train_stream = benchmark.train_stream
test_stream = benchmark.test_stream


# log to Tensorboard
#tb_logger = TensorboardLogger()

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

optimizer = SGD(enc_model.parameters(), lr=args.learning_rate, momentum=0.9)
criterion = CrossEntropyLoss()

if args.replay_strategy == 'replay':
    plugins = [ReplayPlugin(mem_size=args.mem_size)]
elif args.replay_strategy == 'gdumb':
    gdumb = ClassBalancedBuffer(
            max_size=args.mem_size, adaptive_size=True
        )
    plugins = [ReplayPlugin(mem_size=args.mem_size,storage_policy=gdumb)]
else:
    plugins = None
    
encoder_strategy = TrainStrategy(args.strategy_type,
        enc_model, optimizer, criterion, train_epochs=args.epoch, train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
        device=device, plugins=plugins, evaluator=eval_plugin)
# train and test loop over the stream of experiences



# log to text file
#text_logger = TextLogger(open(os.path.join(dpath,'log.txt'), 'a'))
eval_plugin = EvaluationPlugin(
    loss_metrics(minibatch=True, stream=True,experience=True),
    loggers=[InteractiveLogger(), CSVLogger(log_folder=dpath)],
    benchmark = benchmark
)
if args.decoder_strategy == 'replay':
    decoder_strategy = VAEReplayTraining(model, optimizer=Adam(model.decoder.parameters(), lr=args.learning_rate, weight_decay=1e-5),
                          criterion=AE_LOSS,
                          train_epochs=args.epoch, device=device, mem_size=args.mem_size, evaluator=eval_plugin, train_mb_size=args.batch_size,
                          eval_mb_size=args.batch_size)
elif args.decoder_strategy == 'naive':
    decoder_strategy = VAENaiveTraining(model, optimizer=Adam(model.decoder.parameters(), lr=args.learning_rate, weight_decay=1e-5),
                          criterion=AE_LOSS,
                          train_epochs=args.epoch, device=device, evaluator=eval_plugin, train_mb_size=args.batch_size,
                          eval_mb_size=args.batch_size)
else:
    raise NotImplementedError('Not supported type.')


test_stream_2 = copy.deepcopy(test_stream)
for e, (train_exp, test_exp) in enumerate(zip(train_stream, test_stream)):
    train_exp_2 = copy.deepcopy(train_exp)
    torch.nn.init.xavier_uniform_(enc_model.classifier.parameters())
    print("Begin encoder training "+str(e))
    encoder_strategy.train(train_exp)
    encoder_strategy.eval(test_stream)
    print("End encoder training "+str(e))
    for i in enc_model.parameters():
    #    i.requires_grad = False
        print(i[0][1])
        break

    #model.encoder.features = enc_model.features
    #decoder_strategy.model = model

    print("Begin decoder training"+str(e))
    decoder_strategy.train(train_exp_2)
    print("End decoder training" + str(e))

    for i in enc_model.parameters():
    #    i.requires_grad = True
        print(i[0][1])
        break

    representations, images, results, labels = decoder_strategy.eval(test_stream_2)

    labels = labels[-args.T:]
    representations = representations[-args.T:]

    for rp,lbl in zip(representations,labels):
        rp = torch.vstack(rp)
        lbl = torch.concat(lbl)
        representation_log(e+1,rp,lbl,rpath)
    
    before = len(representations)-args.T
    after = len(representations)-args.T+ e+1
    images = images[before: after]
    results = results[before: after]
    images = [x for xs in images for x in xs]
    results = [x for xs in results for x in xs]

    channels, size_x, size_y = results[0].size()[1], results[0].size()[2], results[0].size()[3]
    results=torch.stack(results).view(-1, channels, size_x, size_y)
    images =torch.stack(images).view(-1, channels, size_x, size_y)
    indexes = torch.randperm(results.shape[0])
    results, images = results[indexes], images[indexes]
    image_generator(e+1, images, results,path=dpath,device=args.device)

