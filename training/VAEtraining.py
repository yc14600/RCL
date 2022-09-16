from typing import Optional, Sequence, List, Union

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD

from avalanche.models.pnn import PNN
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.plugins import (
    SupervisedPlugin,
    GEMPlugin,
    CWRStarPlugin,
    ReplayPlugin,
    GenerativeReplayPlugin,
    TrainGeneratorAfterExpPlugin,
    GDumbPlugin,

    AGEMPlugin,
    EvaluationPlugin,
    SynapticIntelligencePlugin,
    CoPEPlugin,
    GSS_greedyPlugin,
    LFLPlugin,
    MASPlugin,
)

from avalanche.training.templates.base import BaseTemplate
from templates.SupervisedTemplate import SupervisedTemplate
from avalanche.models.generator import VAE_loss
from avalanche.logging import InteractiveLogger
from plugins.EWCplugin import EWCPlugin
from plugins.LWFPlugin import LwFPlugin
class VAEEWCTraining(SupervisedTemplate):
    """VAETraining class
    This is the training strategy for the VAE model
    found in the models directory.
    We make use of the SupervisedTemplate, even though technically this is not a
    supervised training. However, this reduces the modification to a minimum.
    We only need to overwrite the criterion function in order to pass all
    necessary variables to the VAE loss function.
    Furthermore we remove all metrics from the evaluator.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        ewc_lambda: float,
        criterion=VAE_loss,
        mode: str = "separate",
        decay_factor: Optional[float] = None,
        keep_importance_data: bool = False,
        mem_size: int = 200,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = EvaluationPlugin(
            loggers=[InteractiveLogger()],
            #suppress_warnings=True,
        ),
        eval_every=-1,
        **base_kwargs
    ):
        """
        Creates an instance of the Naive strategy.
        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param \*\*base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        self.model = model
        rp = ReplayPlugin(mem_size)
        rp2 = EWCPlugin(ewc_lambda, mode, decay_factor, keep_importance_data)
        if plugins is None:
            plugins = [rp]
            plugins.append(rp2)
        else:
            plugins.append(rp)
            plugins.append(rp2)

        super().__init__(
            self.model,
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



    def criterion(self):
        """Adapt input to criterion as needed to compute reconstruction loss
        and KL divergence. See default criterion VAELoss."""
        return self._criterion(self.mb_x, self.mb_output)

class VAELWFTraining(SupervisedTemplate):
    """VAETraining class
    This is the training strategy for the VAE model
    found in the models directory.
    We make use of the SupervisedTemplate, even though technically this is not a
    supervised training. However, this reduces the modification to a minimum.
    We only need to overwrite the criterion function in order to pass all
    necessary variables to the VAE loss function.
    Furthermore we remove all metrics from the evaluator.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        alpha=1,
        temperature=2,
        criterion=VAE_loss,
        mem_size: int = 200,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = EvaluationPlugin(
            loggers=[InteractiveLogger()],
            #suppress_warnings=True,
        ),
        eval_every=-1,
        **base_kwargs
    ):
        """
        Creates an instance of the Naive strategy.
        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param \*\*base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        self.model = model
        rp = ReplayPlugin(mem_size)
        rp2 = LwFPlugin(alpha=alpha, temperature=temperature)
        if plugins is None:
            plugins = [rp]
            plugins.append(rp2)
        else:
            plugins.append(rp)
            plugins.append(rp2)

        super().__init__(
            self.model,
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



    def criterion(self):
        """Adapt input to criterion as needed to compute reconstruction loss
        and KL divergence. See default criterion VAELoss."""
        return self._criterion(self.mb_x, self.mb_output)

class VAEReplayTraining(SupervisedTemplate):
    """VAETraining class
    This is the training strategy for the VAE model
    found in the models directory.
    We make use of the SupervisedTemplate, even though technically this is not a
    supervised training. However, this reduces the modification to a minimum.
    We only need to overwrite the criterion function in order to pass all
    necessary variables to the VAE loss function.
    Furthermore we remove all metrics from the evaluator.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=VAE_loss,
        mem_size: int = 200,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = EvaluationPlugin(
            loggers=[InteractiveLogger()],
            #suppress_warnings=True,
        ),
        eval_every=-1,
        **base_kwargs
    ):
        """
        Creates an instance of the Naive strategy.
        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param \*\*base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        self.model = model
        rp = ReplayPlugin(mem_size)

        if plugins is None:
            plugins = [rp]

        else:
            plugins.append(rp)


        super().__init__(
            self.model,
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



    def criterion(self):
        """Adapt input to criterion as needed to compute reconstruction loss
        and KL divergence. See default criterion VAELoss."""
        return self._criterion(self.mb_x, self.mb_output)

class VAENaiveTraining(SupervisedTemplate):
    """VAETraining class
    This is the training strategy for the VAE model
    found in the models directory.
    We make use of the SupervisedTemplate, even though technically this is not a
    supervised training. However, this reduces the modification to a minimum.
    We only need to overwrite the criterion function in order to pass all
    necessary variables to the VAE loss function.
    Furthermore we remove all metrics from the evaluator.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion=VAE_loss,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = EvaluationPlugin(
            loggers=[InteractiveLogger()],
            #suppress_warnings=True,
        ),
        eval_every=-1,
        **base_kwargs
    ):
        """
        Creates an instance of the Naive strategy.
        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param \*\*base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        self.model = model



        super().__init__(
            self.model,
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



    def criterion(self):
        """Adapt input to criterion as needed to compute reconstruction loss
        and KL divergence. See default criterion VAELoss."""
        return self._criterion(self.mb_x, self.mb_output)

class VAECWRTraining(SupervisedTemplate):
    """VAETraining class
    This is the training strategy for the VAE model
    found in the models directory.
    We make use of the SupervisedTemplate, even though technically this is not a
    supervised training. However, this reduces the modification to a minimum.
    We only need to overwrite the criterion function in order to pass all
    necessary variables to the VAE loss function.
    Furthermore we remove all metrics from the evaluator.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        cwr_layer_name,
        criterion=VAE_loss,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = EvaluationPlugin(
            loggers=[InteractiveLogger()],
            #suppress_warnings=True,
        ),
        eval_every=-1,
        **base_kwargs
    ):
        """
        Creates an instance of the Naive strategy.
        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param \*\*base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        self.model = model
        cwsp = CWRStarPlugin(model, cwr_layer_name, freeze_remaining_model=True)
        if plugins is None:
            plugins = [cwsp]
        else:
            plugins.append(cwsp)



        super().__init__(
            self.model,
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



    def criterion(self):
        """Adapt input to criterion as needed to compute reconstruction loss
        and KL divergence. See default criterion VAELoss."""
        return self._criterion(self.mb_x, self.mb_output)