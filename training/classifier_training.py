from typing import Optional, Sequence, List, Union

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD

from avalanche.models.pnn import PNN
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
                   'agem':AGEMPlugin,'rwalk':RWalkPlugin,'si':SynapticIntelligencePlugin
                   }

stg_params = {'ewc':{'ewc_lambda':0.4,'mode':"online",'decay_factor':0.1,'keep_importance_data':False},
                'lwf':{'alpha':1, 'temperature':2},
                'cwr':{'cwr_layer_name':None},
                'rwalk':{'ewc_lambda':0.1, 'ewc_alpha':0.9, 'delta_t':10},
                'agem':{'patterns_per_experience': 100, 'sample_size': 100},
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

