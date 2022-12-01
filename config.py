import json

DEFAULTS = {
    "size": 380,
    "seed": 4471,
    "optimizer": {
        "batch_size": 32,
        #"type": "SGD",  # supported: SGD, Adam
        #"momentum": 0.9,
        "weight_decay": 0.005,
        #"clip": 1.,
        "lr": 0.1,
        #"classifier_lr": -1,
        #"nesterov": True,
        "schedule": {
            #"type": "constant",  # supported: constant, step, multistep, exponential, linear, poly
            "num_epoch": 5,
            "start_epoch": 0
        }
    },
    #"normalize": {
    #    "mean": [0.485, 0.456, 0.406],
    #    "std": [0.229, 0.224, 0.225]
    #}
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = json.load(fd)
    _merge(defaults, config)
    return config
