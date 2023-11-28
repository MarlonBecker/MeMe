import random
import os

import torch
import numpy as np

from algorithms.base import BaseTrainer
from utils.parameterScheduler import get_scheduler
from models import get_model


class AbsWithGradient(torch.autograd.Function):
    """
    wrapper around torch.abs which has gradient of 1 at input 0 instead of 0
    """
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return torch.abs(input_)

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        return grad_output * (1 - 2 * (input_ < 0))

absWithGradient = AbsWithGradient.apply


def softplus_normed(x: np.ndarray):
    # thanks to https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
    x = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    return x / np.sum(x)

class MeMeTrainer(BaseTrainer):
    identifier = "meme"
    def __init__(self, config):
        super(MeMeTrainer, self).__init__(config)

        self.revert = False
        self.reward = 0
        self.action = 0
        self.loss = 0
        self.powers = None
        self.overlaps = None

        self.metropolis_temperature = get_scheduler(self.config["algorithm"]["metropolis_temperature"], self.config)
        self.sampling_temperature = get_scheduler(self.config["algorithm"]["sampling_temperature"], self.config)

        self.model = get_model(config, action_space_shape = self.env.get_action_space_shape())
        self.model_output: torch.Tensor = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["algorithm"]["learning_rate"], momentum = 0, weight_decay = 0)

        if self.config["algorithm"]["structureInitThresholdNet"]:
            self.model_output = self.model.forward()
            structure = (self.model_output.detach().numpy() < 0).reshape(self.env.env.designArea)
            self.env.reset(reset_structure=structure)

        self.netOutputLimit = None
        if self.config["algorithm"]["netOutputLimit"] is not None:
            self.netOutputLimit = self.config["algorithm"]["netOutputLimit"] * self.sampling_temperature.baseValue

        self.dataLogger.after_init(trainer=self)

        if self.config["existingFileMode"] == "continue":
            self.loadCheckpoint()


    def get_TOM_and_actions(self, num_actions: int = 1):
        self.model_output = self.model.forward()

        # template matching operation
        TOM_out: torch.Tensor = self.model_output * (-1 + 2 * torch.tensor(self.env.get_optimizable_structure().flatten()))

        # apply soft limit to net output
        if self.netOutputLimit is not None:
            # using: (1 - 2 * (TOM_out > 0)) == -sign(efficiency_change_prediction) but with sign(0) = 1 instead of 0
            TOM_out = (1 - 2 * (TOM_out < 0)) * self.netOutputLimit * (1 -torch.exp(-absWithGradient(TOM_out)/self.netOutputLimit))

        expectedReward = TOM_out.detach().numpy()

        # store/calculate efficiency buffer part and structure reg part of loss function separately to be able to apply reg factor scheduling
        if self.struct_reg_factor is not None:
            reg_change = self.calc_struct_regularizer(self.env.structure, returnChangeMatrix=True).flatten()
            expectedReward -= self.struct_reg_factor(self.iteration) * reg_change

        if self.sampling_temperature(self.iteration) == 0:
            if num_actions == 1:
                actions = np.argmax(expectedReward)
            else:
                actions = np.argpartition(expectedReward, -num_actions)[-num_actions:]
        elif np.isinf(self.sampling_temperature(self.iteration)):
            actions = np.random.choice(self.action_space_size, size = num_actions , replace = False)
        else:
            p_ = softplus_normed(expectedReward/self.sampling_temperature(self.iteration))
            if np.sum(p_ != 0) == 0:
                if num_actions == 1:
                    actions = np.argmax(expectedReward)
                else:
                    actions = np.argpartition(expectedReward, -num_actions)[-num_actions:]
                print(">>>> TOO MANY NON ZERO ENTRIES IN SAMPLING DISTRIBUTION -> fall back to argmax before softmax")
            else:
                actions = np.random.choice(self.action_space_size, size = num_actions , replace = False, p = p_)
        return actions, TOM_out

    def train(self):
        if self.struct_reg_factor is not None:
            last_struct_reg = self.calc_struct_regularizer(self.env.structure)

        while self.iteration < self.iterations:
            # evaluate net and sample action
            actions, TOM_out = self.get_TOM_and_actions()
            self.action = actions[0]

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

            self.loss = self.calculateLoss([(self.action, self.reward),], TOM_out)
            self.optimizer.zero_grad()

            if "outputGradients" in self.config["metrics"]:
                self.model_output.retain_grad()
            self.loss.backward()
            self.optimizer.step()

            self.iteration+=1
            self.dataLogger.on_training_step(trainer = self, iteration = self.iteration)
            self.dataLogger.on_batch_end(trainer = self, iteration = self.iteration)

        self.dataLogger.on_training_end(trainer=self,)


    def calculateLoss(self, results, predictions: torch.Tensor):
        if self.config["algorithm"]["loss"] == "squared_error":
            loss = torch.zeros((1,))
            for act, res in results:
                loss += 0.5*(predictions[act]-res)**2
            return loss
        if self.config["algorithm"]["loss"] == "result_product":
            loss = torch.zeros((1,))
            for act, res in results:
                # print(predictions[act], res)
                loss += -predictions[act]*res
            return loss
        if self.config["algorithm"]["loss"] == "result_product_damped":
            loss = torch.zeros((1,))
            for act, res in results:
                loss += -torch.sign(predictions[act]*res) * torch.log(torch.abs(predictions[act]*res) + 1)
            return loss
        if self.config["algorithm"]["loss"] == "result_product_limted":
            limit = self.config["algorithm"]["result_product_limted_limit"]
            loss = torch.zeros((1,))
            for act, res in results:
                # special case handled separately since torch.abs kills all gradients at x = 0
                if predictions[act]*res == 0:
                    loss += -predictions[act]*res
                else:
                    loss += -torch.sign(predictions[act]*res) * limit * (1 -torch.exp(-torch.abs(predictions[act]*res)/limit))
            return loss

    def saveCheckpoint(self, checkpointPath) -> None:
        super(MeMeTrainer, self).saveCheckpoint(checkpointPath)

        torch.save(self.model.state_dict(), os.path.join(checkpointPath, f"net.model"))

    def loadCheckpoint(self) -> None:
        super(MeMeTrainer, self).loadCheckpoint()

        net_state_dict = torch.load(os.path.join(checkpointPath, f"net.model"))
        self.model.load_state_dict(net_state_dict)
