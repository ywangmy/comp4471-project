import torch
from omegaconf import OmegaConf

def main(conf: OmegaConf):
    try:
        conf = OmegaConf.merge(conf, OmegaConf.load(conf.conf_file))
    except:
        pass
    OmegaConf.set_readonly(conf, True)
    print(OmegaConf.to_yaml(conf))

# python test.py conf_file=./exps/exp1.yaml
if __name__ == '__main__':
    main(OmegaConf.from_cli())
