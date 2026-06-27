from pathlib import Path
import yaml


class Config:

    def __init__(self, path: str):
        with open(path, "r") as f:
            self.config = yaml.safe_load(f)

    def __getitem__(self, key):
        return self.config[key]