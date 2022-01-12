from omegaconf import DictConfig, OmegaConf
import hydra
import os
import json

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()