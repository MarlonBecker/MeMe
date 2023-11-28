import numpy as np


available_schedulers = {}
def addScheduler(class_):
    if class_.name is None:
        raise ValueError(f"Scheduler.name has to be defined in metric class definition for metric {class_}.")
    available_schedulers[class_.name] = class_
    return class_

@addScheduler
class ConstScheduler:
    name: str = "const"
    def __init__(self, baseValue: float, maxLength: float, scheduler_config: dict = None) -> None:
        self.baseValue = baseValue

    def __call__(self, progress: float) -> float:
        return self.baseValue

@addScheduler
class CosineDecayScheduler:
    name: str = "cosDecay"
    def __init__(self, baseValue: float, maxLength: float, scheduler_config: dict = None) -> None:
        self.baseValue = baseValue
        self.maxLength = maxLength

    def __call__(self, progress: float) -> float:
        return self.baseValue * 0.5 * (1 + np.cos((progress/self.maxLength) * np.pi))

@addScheduler
class CosineIncreaseScheduler:
    name: str = "cosIncrease"
    def __init__(self, baseValue: float, maxLength: float, scheduler_config: dict = None) -> None:
        self.baseValue = baseValue
        self.maxLength = maxLength

    def __call__(self, progress: float) -> float:
        return self.baseValue * 0.5 * (1 - np.cos((progress/self.maxLength) * np.pi))

@addScheduler
class ExpIncreaseScheduler:
    name: str = "expIncrease"
    def __init__(self, baseValue: float, maxLength: float, scheduler_config: dict = None) -> None:
        if scheduler_config is None or "rate" not in scheduler_config:
            self.rate = 1e-4
        else:
            self.rate = scheduler_config["rate"]

        self.baseValue = baseValue

    def __call__(self, progress: float) -> float:
        return self.baseValue * np.exp(progress * self.rate)


def get_scheduler(scheduler_arg_list, config: dict):
    if scheduler_arg_list is None:
        return None
    if not isinstance(scheduler_arg_list, list):
        return available_schedulers["const"](scheduler_arg_list, config["algorithm"]["iterations"])

    assert len(scheduler_arg_list) <= 3, "allowed parameter inputs:\n value \n [scheduler_name, value]\n [scheduler_name, value, scheduler_config_dict]"

    name = scheduler_arg_list[0]
    baseValue = scheduler_arg_list[1]
    scheduler_config = scheduler_arg_list[2] if len(scheduler_arg_list) == 3 else None

    if name not in available_schedulers:
        raise RuntimeError(f"Scheduler '{name}' not found. Availabe: {', '.join(available_schedulers.keys())}")

    return available_schedulers[name](baseValue, config["algorithm"]["iterations"], scheduler_config)
