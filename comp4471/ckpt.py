import os
import shutil
import torch

class State:
    def __init__(self, model, optimizer, ckpt_path: str = None):
        self.epoch = 0
        self.iter = 0
        self.min = None
        self.model = model
        self.optimizer = optimizer
        self.ckpt_path = ckpt_path

    def capture_snapshot(self):
        return {
            "epoch": self.epoch,
            "iter": self.iter,
            "min": self.min,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def apply_snapshot(self, obj):
        self.epoch = obj["epoch"]
        self.iter = obj["iter"]
        self.min = obj["min"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])

    def save(self, metric=None, filename=None):
        if filename is None: filename = self.ckpt_path

        ckpt_dir = os.path.dirname(filename)
        os.makedirs(ckpt_dir, exist_ok=True)

        if metric is not None and (self.min is None or self.min > metric):
            self.min = metric
            best = True
        else: best = False

        # save to tmp, then commit by moving the file in case the job gets interrupted while writing the checkpoint
        tmp_filename = filename + ".tmp"
        torch.save(self.capture_snapshot(), tmp_filename)
        os.rename(tmp_filename, filename)
        print(f"=> saved checkpoint for epoch {self.epoch} at {filename}")

        if best:
            best_filename = filename + ".best"
            shutil.copyfile(filename, best_filename)
            print(f"=> best model found at epoch {self.epoch} saved to {best_filename}")

    def load(self, device, filename=None):
        if filename is None: filename = self.ckpt_path
        if os.path.isfile(self.ckpt_path) is False: return
        # Map model to be loaded to specified single gpu.
        self.apply_snapshot(torch.load(filename, map_location=device))
        print(f'ckpt found, start with epoch={self.epoch}, iter={self.iter}, min={self.min}')