import multiprocessing as mp
import os
import shutil
import pickle

import torch # just to check gpu number
import numpy as np

from utils.env_wrapper import SimEnv
from utils.dataLogger import DataLogger
from utils.structure_regularizer import get_struct_regularizer
from utils.parameterScheduler import get_scheduler

QUEUE_STOP = "QUEUE_STOP"

def setCudaDevice(workerIndex: int) -> None:
    num_available_gpus = int(os.environ.get("X_CUDA_VISIBLE_DEVICES", torch.cuda.device_count()))
    if num_available_gpus == 0:
        print("No GPUs available")
    else:
        device = workerIndex % num_available_gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
        print(f"Available GPUs: {num_available_gpus}, worker_index: {workerIndex}, setting CUDA_VISIBLE_DEVICES: {device}")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

class InheritanceTracerMetaClass(type):
    """
    used to track which classes inherit from BaseTrainer to auto-add them to dict and thus make selectable via config arguments
    (build in __subclasses__ does not contain indirect child classes)
    """
    child_classes = []
    def __new__(metaCls, *args, **kwargs):
        child_class = type.__new__(metaCls, *args, **kwargs)
        metaCls.child_classes.append(child_class)
        return child_class

class BaseTrainer(metaclass = InheritanceTracerMetaClass):
    """
    new trainers have to inherit from this class!!!
    """
    def __init__(self, config,):
        self.config = config

        #set env
        reset_structure = np.loadtxt(config["algorithm"]["structureFilePath"], delimiter = ",", dtype = int) if config["algorithm"]["structureFilePath"] is not None else None
        self.env = SimEnv(self.config["env_config"], reset_structure=reset_structure)
        self.efficiency = self.env.efficiency
        self.last_efficiency = self.efficiency
        self.action_space_size = self.env.get_action_space_size()

        self.dataLogger = DataLogger(config = self.config)

        self.iteration = 0
        self.iterations = self.config["algorithm"]["iterations"]

        self.struct_reg = 0
        self.struct_reg_factor = get_scheduler(self.config["algorithm"]["struct_reg_factor"], self.config)
        if self.struct_reg_factor is not None:
            self.calc_struct_regularizer = get_struct_regularizer(self.config["algorithm"]["struct_reg_identifier"])
        else:
            self.calc_struct_regularizer = None

        setCudaDevice(workerIndex = 0)

    def check_for_checkpoint_saving(self) -> None:
        if self.config["checkpointInterval"] == 0 or self.config["checkpointInterval"] == None:
            return
        if self.iteration % self.config["checkpointInterval"] == 0:
            checkpointPath = os.path.join(self.config["logDir"], "checkpoint")
            bu_checkpointPath = os.path.join(self.config["logDir"], "checkpoint_bu")

            if os.path.isdir(checkpointPath):
                shutil.move(checkpointPath, bu_checkpointPath)

            self.saveCheckpoint(checkpointPath)

            if os.path.isdir(bu_checkpointPath):
                shutil.rmtree(bu_checkpointPath)

    def saveCheckpoint(self, checkpointPath) -> None:
        os.makedirs(checkpointPath, exist_ok=False)
        state = {
            "iteration": self.iteration,
            "structure": self.env.structure,
        }

        with open(os.path.join(checkpointPath, 'state.pickle'), 'wb') as f:
            pickle.dump(state, f)

    def loadCheckpoint(self) -> None:
        checkpointPath = os.path.join(self.config["logDir"], "checkpoint")
        if not os.path.isdir(checkpointPath):
            return

        with open(os.path.join(checkpointPath, 'state.pickle'), 'rb') as f:
            state = pickle.load(f)

        self.iteration = state["iteration"]
        if self.iteration == self.iterations:
            raise RuntimeError(f"Running in continue mode and found checkpoint for iteration {self.iteration} which equals max iteration. Use '--existingFileMode truncate' to restart run.")
        self.env.reset(reset_structure=state["structure"])
        self.efficiency = self.env.efficiency
        self.last_efficiency = self.efficiency

        print(f"Model and data found! Starting from iteration {self.iteration}")

    def train(self):
        raise NotImplementedError


class ParallelExplorationWorker():
    """get actions and structure and return efficiencies"""
    def __init__(self, config, worker_index, jobQueue, resultQueue):
        """
        new init since BaseWorker i.a. initializes one callback per worker
        """
        self.config = config
        self.config["env_config"]["worker_index"] = worker_index

        #set env
        self.env = SimEnv(self.config["env_config"])
        self.efficiency = self.env.efficiency

        self.jobQueue = jobQueue
        self.resultQueue = resultQueue


        setCudaDevice(workerIndex = worker_index)


    def train(self):
        while True:
            #fetch action and structure
            job = self.jobQueue.get()
            if job == QUEUE_STOP:
                return

            action, structure = job
            self.env.setStructure(structure)
            efficiency, overlaps, powers = self.env.step(action)

            self.resultQueue.put((action, efficiency, overlaps, powers))

class ParallelTrainer(BaseTrainer):
    identifier = None
    ParallelExplorationWorkerClass = ParallelExplorationWorker

    def __init__(self, config):
        super(ParallelTrainer, self).__init__(config)
        self.N_worker = self.config["algorithm"]["num_workers"]

        self.jobQueue = mp.Queue()
        self.resultQueue = mp.Queue()

    def startSlaveWorker(self, index):
        try:
            worker = self.ParallelExplorationWorkerClass(self.config, index, self.jobQueue, self.resultQueue)
            worker.train()
        finally:
            # make sure to tell master to stop if an error occurred in worker
            self.resultQueue.put(QUEUE_STOP)

    def train(self) -> None:
        print("parallel train running")
        
        slaveProcesses = [mp.Process(target = self.startSlaveWorker, args=(index+1, )) for index in range(self.N_worker)]
        for p in slaveProcesses:
            p.start()

        try:
            self.master_train()
        finally:
            # make sure to tell all workers to stop if an error occurred in master
            for _ in range(self.N_worker):
                self.jobQueue.put(QUEUE_STOP)

        for p in slaveProcesses:
            p.join()

    def master_train(self) -> None:
        raise NotImplementedError

    def __del__(self):
        #stop workers
        for _ in range(self.N_worker):
            self.jobQueue.put(QUEUE_STOP)
