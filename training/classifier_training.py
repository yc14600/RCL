from typing import Optional, Sequence, List, Union
from pkg_resources import parse_version

import torch
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from collections import defaultdict

from avalanche.models.pnn import PNN
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import (
    SupervisedPlugin,
    CWRStarPlugin,
    ReplayPlugin,
    GenerativeReplayPlugin,
    TrainGeneratorAfterExpPlugin,
    GDumbPlugin,
    LwFPlugin,
    AGEMPlugin,
    GEMPlugin,
    EWCPlugin,
    EvaluationPlugin,
    SynapticIntelligencePlugin,
    CoPEPlugin,
    GSS_greedyPlugin,
    LFLPlugin,
    MASPlugin,
    RWalkPlugin,
)

from avalanche.training.templates.supervised import SupervisedTemplate


train_strategies ={'ewc':EWCPlugin,'cwr':CWRStarPlugin,'lwf':LwFPlugin,
                   'agem':AGEMPlugin,'rwalk':RWalkPlugin,'si':SynapticIntelligencePlugin,
                   'gem':GEMPlugin,
                   }

stg_params = {'ewc':{'ewc_lambda':0.4,'mode':"online",'decay_factor':0.1,'keep_importance_data':False},
                'lwf':{'alpha':1, 'temperature':2},
                'cwr':{'cwr_layer_name':None},
                'rwalk':{'ewc_lambda':0.01, 'ewc_alpha':0.99, 'delta_t':10},
                'agem':{'patterns_per_experience': 200, 'sample_size': 100},
                'gem':{'patterns_per_experience':400, 'memory_strength':0.8},
                'si':{'si_lambda':0.0001}
                }



class BaseStrategy(SupervisedTemplate):
        def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            criterion=CrossEntropyLoss(),
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: Optional[int] = None,
            device=None,
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator: EvaluationPlugin = default_evaluator,
            eval_every=-1,
            **base_kwargs
        ):
            super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

        def reset_optimizer(optimizer, model):
            """Reset the optimizer to update the list of learnable parameters.
            .. warning::
                This function fails if the optimizer uses multiple parameter groups.
            :param optimizer:
            :param model:
            :return:
            """
            assert len(optimizer.param_groups) == 1
            optimizer.state = defaultdict(dict)

class TrainStrategy(BaseStrategy):
        def __init__(
            self,
            strategy: str,
            model: Module,
            optimizer: Optimizer,
            criterion=CrossEntropyLoss(),
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: Optional[int] = None,
            device=None,
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator: EvaluationPlugin = default_evaluator,
            eval_every=-1,
            **base_kwargs
        ):
            if strategy != 'naive':
                if strategy == 'cwr':
                    stg_params[strategy].update({'model':model})
                splugin = train_strategies[strategy](**stg_params[strategy])
                if plugins is None:
                    plugins=[splugin]
                else:
                    plugins.append(splugin)
                
            super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
        def make_train_dataloader(
                self,
                num_workers=0,
                shuffle=True,
                pin_memory=False,
                persistent_workers=False,
                **kwargs
        ):
            """Data loader initialization.
            Called at the start of each learning experience after the dataset
            adaptation.
            :param num_workers: number of thread workers for the data loading.
            :param shuffle: True if the data should be shuffled, False otherwise.
            :param pin_memory: If True, the data loader will copy Tensors into CUDA
                pinned memory before returning them. Defaults to True.
            """

            other_dataloader_args = {}

            if parse_version(torch.__version__) >= parse_version("1.7.0"):
                other_dataloader_args["persistent_workers"] = persistent_workers

            self.dataloader = TaskBalancedDataLoader(
                self.adapted_dataset,
                oversample_small_groups=True,
                num_workers=num_workers,
                batch_size=self.train_mb_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                **other_dataloader_args
            )

        def make_eval_dataloader(
                self, num_workers=0, pin_memory=False, persistent_workers=False, **kwargs
        ):
            """
            Initializes the eval data loader.
            :param num_workers: How many subprocesses to use for data loading.
                0 means that the data will be loaded in the main process.
                (default: 0).
            :param pin_memory: If True, the data loader will copy Tensors into CUDA
                pinned memory before returning them. Defaults to True.
            :param kwargs:
            :return:
            """
            other_dataloader_args = {}

            if parse_version(torch.__version__) >= parse_version("1.7.0"):
                other_dataloader_args["persistent_workers"] = persistent_workers

            self.dataloader = DataLoader(
                self.adapted_dataset,
                num_workers=num_workers,
                batch_size=self.eval_mb_size,
                pin_memory=pin_memory,
                **other_dataloader_args
            )
