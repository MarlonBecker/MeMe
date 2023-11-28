import random
import numpy as np

from algorithms.base import ParallelTrainer, QUEUE_STOP
from algorithms.metropolis import MetropolisTrainer

class ParallelMetropolisTrainer(ParallelTrainer, MetropolisTrainer):
    identifier = "parallelMetropolis"

    def master_train(self):
        if self.struct_reg_factor is not None:
            last_struct_reg = self.calc_struct_regularizer(self.env.structure)

        num_actions_accepted = 0
        only_accepted_efficiency = None # if only one actions is accepted, this is used as buffer for the efficiency and recalculation can be skipped

        stopMainLoop = False
        while not stopMainLoop:
            # evaluate net and sample action
            actions = np.random.choice(self.action_space_size, size = self.N_worker , replace = False)

            # distribute jobs to workers through queue
            for action in actions:
                self.jobQueue.put((action, self.env.structure))

            # if more than 1 flip was accepted: update current state while workers are simulating next iteration
            if num_actions_accepted > 1:
                self.last_efficiency, _, _, _ = self.env.evaluate()
            # if 1 flip was accepted: directly copy the corresponding efficiency without reevaluating
            if num_actions_accepted == 1:
                self.last_efficiency = only_accepted_efficiency
            # else: no updated needed

            # fetch worker results and perform actions (steps)
            num_actions_accepted = 0
            for _ in range(self.N_worker):
                # worker results are written to self.XX to let metrics fetch them after step
                result = self.resultQueue.get()
                if result == QUEUE_STOP: #should only occur if an error was raised in worker
                    return
                self.action, self.efficiency, self.overlaps, self.powers = result
                # beware! this efficiency does not directly correspond to the current structure due to parallel workers evaluation of single actions. you should keep this in mind if reconstructing the structure from actions/reverts in plotting scripts

                self.reward = self.efficiency - self.last_efficiency

                # performing action is necessary for fab const reg calculation
                # revert simply performs action without computing efficiency again 
                self.env.revert(self.action)

                # calc and add structure reg factor
                if self.struct_reg_factor is not None:
                    self.struct_reg = self.calc_struct_regularizer(self.env.structure)
                    self.reward -= self.struct_reg_factor(self.iteration)*(self.struct_reg-last_struct_reg)

                self.revert = random.random() >= np.exp(self.reward / self.metropolis_temperature(self.iteration))
                if self.revert:
                    self.env.revert(self.action)
                else:
                    only_accepted_efficiency = self.efficiency
                    num_actions_accepted += 1
                    if self.struct_reg_factor is not None:
                        last_struct_reg = self.struct_reg

                self.iteration+=1
                self.dataLogger.on_training_step(trainer = self, iteration = self.iteration)
                if self.iteration >= self.iterations:
                    stopMainLoop = True
                    break

            self.dataLogger.on_batch_end(trainer = self, iteration = self.iteration)

        self.dataLogger.on_training_end(trainer=self,)
