import warnings
from typing import Tuple
import time
import os

import numpy as np
import h5py

class DataLogger():
    print_frequency: int = 20
    def __init__(self, config):
        self.config = config
        # add metrics
        self.metrics: list[BaseMetric] = []
        for metricName in config["metrics"]:
            if metricName in available_metrics:
                self.metrics.append(available_metrics[metricName](config = config))
            else:
                raise RuntimeError(f"Metric {metricName} not found. Available metrics: {', '.join(available_metrics.keys())}")

        self.startTime = time.time()

    def write_data(self, dataName: str, data: [int, float, np.ndarray]):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        fileName = os.path.join(self.config["logDir"], f"data.hdf5")
        with h5py.File(fileName, "a") as file:
            file.create_dataset(dataName, shape=data.shape if data.size>1 else (1,) , dtype=data.dtype, chunks=True if data.size>1 else None)
            file[dataName][:] = data

    def after_init(self, trainer):
        for metric in self.metrics:
            metric.after_init(trainer = trainer)

    def on_training_step(self, trainer, iteration: int):
        if iteration % self.print_frequency == 0:
            delta_time = time.time()-self.startTime
            print(f"Steps done: {iteration}. Current Efficiency: {trainer.last_efficiency}. {self.print_frequency/delta_time:.3f} iterations per second.")
            self.startTime = time.time()

        trainer.check_for_checkpoint_saving()
        for metric in self.metrics:
            metric.on_training_step(iteration = iteration, trainer = trainer)

    def on_batch_end(self, trainer, iteration: int):
        for metric in self.metrics:
            metric.on_batch_end(iteration = iteration, trainer = trainer)

    def on_training_end(self, trainer):
        for metric in self.metrics:
            metric.on_training_end(trainer = trainer)

class BaseMetric:
    name: str = None
    logEach: int = 1
    flushEach: int = 20
    shape: Tuple[int] = () #defines shape of data (excluding length/batch/epoch dimension)
    maxIterations: int = None
    dtype = float
    def __init__(self, config):
        self.buffer = []
        
        self.fileName = os.path.join(config["logDir"], f"data.hdf5")

        with h5py.File(self.fileName, "a") as file:
            if self.name not in file:
                file.create_dataset(self.name, shape=(0, *self.shape), dtype=self.dtype, maxshape=(self.maxIterations, *self.shape), chunks=True)

    def after_init(self, trainer):
        pass

    def on_training_step(self, iteration: int, trainer):
        if iteration % self.logEach == 0:
            metric_data: np.ndarray = self.calcMetric(trainer)
            self.buffer.append(metric_data)
            
        if iteration % self.flushEach == 0:
            self.flush()

    def on_batch_end(self, iteration: int, trainer):
        pass

    def on_training_end(self, trainer):
        """
        """
        if len(self.buffer) == 0:
            return
        
        self.flush()

    def calcMetric(self, trainer):
        """
        calcMetric should calculate metric from 'trainer', convert to np.ndarray, and make it ready for reduction by _reduceData
        return: np.array
        """
        raise NotImplementedError

    def _reduceData(self):
        """reduce data to write to buffer
        default: no reduction
        
        return np.array or convertible to np.array which has shape [dataLen, self.shape], where dataLen is concatenated into file
        return values have to have at least 1 dim (i.e. len(...)==1)
        """
        return np.concatenate(self.buffer)


    def flush(self):
        if len(self.buffer) == 0:
            warnings.warn(f"flush() called, but no data collected for metric {self.name}.")
            return

        with h5py.File(self.fileName, "r+") as file:
            data = self._reduceData()

            dataset = file[self.name]
            dataset.resize(len(dataset)+len(data), axis = 0)
            dataset[len(dataset)-len(data):, ...] = np.array(data)

            self.buffer = []



available_metrics = {}
def addMetric(class_): #@TODO add metrics by checking inheritance
    if class_.name is None:
        raise ValueError(f"Metric.name has to be defined in metric class definition for metric {class_}.")
    available_metrics[class_.name] = class_
    return class_


"""
 - to create new metrics, simply define classes here which inherit from BaseMetric and decorate with @addMetric
 - overwrite calcMetric and other needed functions (see definitions above)
"""

@addMetric
class EfficiencyMetric(BaseMetric):
    name = "efficiency"
    def calcMetric(self, trainer):
        return [trainer.efficiency]
    
@addMetric
class RewardMetric(BaseMetric):
    name = "reward"
    def calcMetric(self, trainer):
        return [trainer.reward]

@addMetric
class ActionMetric(BaseMetric):
    name = "action"
    dtype = int
    def calcMetric(self, trainer):
        return [trainer.action]

@addMetric
class RevertMetric(BaseMetric):
    name = "revert"
    dtype = bool
    def calcMetric(self, trainer):
        return [trainer.revert]

@addMetric
class LossMetric(BaseMetric):
    name = "loss"
    def calcMetric(self, trainer):
        return [trainer.loss]

@addMetric
class TimeMetric(BaseMetric):
    name = "time"
    def __init__(self, config):
        super(TimeMetric, self).__init__(config)
        self.startTime = time.time()
        
    def calcMetric(self, trainer):
        return [time.time()-self.startTime]

@addMetric
class StructRegMetric(BaseMetric):
    name = "structReg"
    def calcMetric(self, trainer):
        return [trainer.struct_reg]

@addMetric
class PowersMetric(BaseMetric):
    name = "powers"
    firstCall = True
    
    def calcMetric(self, trainer):
        # set shape dynamically at firs call. @TODO determine powers shape from input file and set in init (cf. 'initialStructure')
        if self.firstCall:
            if len(trainer.powers) == 0:
                raise RuntimeError("Metirc 'powers' selected, but there are no powers!")
            self.shape = len(trainer.powers), len(trainer.powers[0])
            with h5py.File(self.fileName, "a") as file:
                del file[self.name]
                file.create_dataset(self.name, shape=(0, *self.shape), dtype=self.dtype, maxshape=(self.maxIterations, *self.shape), chunks=True)
            self.firstCall = False

        return [trainer.powers]

@addMetric
class OverlapsMetric(BaseMetric):
    name = "overlaps"
    firstCall = True
    
    def calcMetric(self, trainer):
        # set shape dynamically at firs call. @TODO determine overlaps shape from input file and set in init (cf. 'initialStructure')
        if self.firstCall:
            if len(trainer.overlaps) == 0:
                raise RuntimeError("Metirc 'overlaps' selected, but there are no overlaps!")
            self.shape = len(trainer.overlaps), len(trainer.overlaps[0])
            with h5py.File(self.fileName, "a") as file:
                del file[self.name]
                file.create_dataset(self.name, shape=(0, *self.shape), dtype=self.dtype, maxshape=(self.maxIterations, *self.shape), chunks=True)
            self.firstCall = False

        return [trainer.overlaps]

@addMetric
class InitialStructureMetric(BaseMetric):
    name = "initialStructure"
    
    #redefine init to set shape at runtime
    def __init__(self, config):
        self.shape = config["env_config"]["environment"]["designArea"]
        super(InitialStructureMetric, self).__init__(config)
    
    def on_training_step(self, iteration: int, trainer):
        pass

    def after_init(self, trainer):
        with h5py.File(self.fileName, "r+") as file:
            dataset = file[self.name]
            dataset.resize(len(dataset) + 1, axis = 0)
            dataset[len(dataset)-1, ...] = trainer.env.structure

    def on_training_end(self, trainer):
        pass


@addMetric
class NetworkOutputMetric(BaseMetric):
    name = "networkOutput"
    #redefine init to set shape and logEach at runtime
    def __init__(self, config):
        self.shape = config["env_config"]["environment"]["designArea"]
        if "num_workers" in config["algorithm"]:
            self.flushEach = max(self.flushEach, config["algorithm"]["num_workers"])
        self.lastFlustIteration = 0
        super(NetworkOutputMetric, self).__init__(config)

    def on_training_step(self, iteration: int, trainer):
        pass

    def on_batch_end(self, iteration: int, trainer):
        if iteration % self.logEach == 0:
            metric_data: np.ndarray = self.calcMetric(trainer)
            self.buffer.append(metric_data)

        if iteration - self.lastFlustIteration >= self.flushEach:
            self.lastFlustIteration = iteration
            self.flush()

    def calcMetric(self, trainer):
        return [trainer.model_output.detach().numpy().reshape(self.shape)]

@addMetric
class OutputGradientsMetric(BaseMetric):
    """
    only works for single layer net atm
    """
    name = "outputGradients"

    #redefine init to set shape and logEach at runtime
    def __init__(self, config):
        self.shape = config["env_config"]["environment"]["designArea"]
        if "num_workers" in config["algorithm"]:
            self.flushEach = max(self.flushEach, config["algorithm"]["num_workers"])
        self.lastFlustIteration = 0
        super(OutputGradientsMetric, self).__init__(config)

    def on_training_step(self, iteration: int, trainer):
        pass

    def on_batch_end(self, iteration: int, trainer):
        if iteration % self.logEach == 0:
            metric_data: np.ndarray = self.calcMetric(trainer)
            self.buffer.append(metric_data)

        if iteration - self.lastFlustIteration >= self.flushEach:
            self.lastFlustIteration = iteration
            self.flush()

    def calcMetric(self, trainer):
        return [trainer.model_output.grad.detach().numpy().reshape(self.shape)]

@addMetric
class InitialEfficiencyMetric(BaseMetric):
    name = "initialEfficiency"
    
    def on_training_step(self, iteration: int, trainer):
        pass

    def after_init(self, trainer):
        with h5py.File(self.fileName, "r+") as file:
            dataset = file[self.name]
            dataset.resize(len(dataset) + 1, axis = 0)
            dataset[len(dataset)-1, ...] = trainer.efficiency

    def on_training_end(self, trainer):
        pass
