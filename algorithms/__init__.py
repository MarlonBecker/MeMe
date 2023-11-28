from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f)]
__all__.remove("__init__")

from . import *

def get_trainer(config):
    """
    collect all classes which inherit from BaseTrainer and return selected by 'algorithm/identifier' key in config
    """
    availabeTrainers = {}
    for cls_ in base.InheritanceTracerMetaClass.child_classes:
        if cls_ is not base.BaseTrainer:
            identifier = cls_.identifier or cls_.__name__
            availabeTrainers[identifier] = cls_

    algorithm_id = config["algorithm"]["identifier"]
    if algorithm_id in availabeTrainers:
        return availabeTrainers[algorithm_id](config = config)
    else:
        raise RuntimeError(f"Trainer '{algorithm_id}' not found. Available trainers: {', '.join(availabeTrainers.keys())}")