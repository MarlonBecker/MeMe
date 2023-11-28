from models.locally_connected import LocallyConnectedNet

modelDictionary = {
    "LocallyConnectedNet": LocallyConnectedNet,
}

def get_model(config, action_space_shape):
    """
    return selected model by 'algorithm/model/identifier' key in config
    """
    model_id = config["algorithm"]["model"]["identifier"]
    if model_id in modelDictionary:
        return modelDictionary[model_id](action_space_shape = action_space_shape, model_config = config["algorithm"]["model"])
    else:
        raise RuntimeError(f"Model '{model_id}' not found. Available models: {', '.join(modelDictionary.keys())}")
