import random

import numpy as np

from algorithms.base import BaseTrainer
from utils.parameterScheduler import get_scheduler


class MetropolisTrainer(BaseTrainer):
    identifier = "metropolis"
    def __init__(self, config):
        super(MetropolisTrainer, self).__init__(config)

        self.revert = False
        self.reward = 0
        self.action = 0
        self.powers = None
        self.overlaps = None

        self.metropolis_temperature = get_scheduler(self.config["algorithm"]["metropolis_temperature"], self.config)

        self.dataLogger.after_init(trainer=self)

        if self.config["existingFileMode"] == "continue":
            self.loadCheckpoint()

    def train(self):
        if self.struct_reg_factor is not None:
            last_struct_reg = self.calc_struct_regularizer(self.env.structure)

        while self.iteration < self.iterations:
            self.action = random.randrange(self.action_space_size)

            # perform action
            self.efficiency, self.overlaps, self.powers = self.env.step(self.action)
            self.reward = self.efficiency - self.last_efficiency

            # calc and add structure reg factor
            if self.struct_reg_factor is not None:
                self.struct_reg = self.calc_struct_regularizer(self.env.structure)
                self.reward -= self.struct_reg_factor(self.iteration)*(self.struct_reg-last_struct_reg)

            self.revert = random.random() >= np.exp(self.reward / self.metropolis_temperature(self.iteration))
            if self.revert:
                self.env.revert(self.action)
            else:
                self.last_efficiency = self.efficiency
                if self.struct_reg_factor is not None:
                    last_struct_reg = self.struct_reg

            self.iteration+=1
            self.dataLogger.on_training_step(trainer = self, iteration = self.iteration)
            self.dataLogger.on_batch_end(trainer = self, iteration = self.iteration)

        self.dataLogger.on_training_end(trainer=self,)
