import logging
import resource
import torch
from abc import abstractmethod
from omegaconf import DictConfig


class BaseExperiment:

    def __init__(self, cfg: DictConfig, exp_dir: str):

        self.log = logging.getLogger("Experiment")
        self.log.info(f"Initializing {self.__class__.__name__}")

        self.cfg = cfg
        self.exp_dir = exp_dir
        self.device = torch.device(
            f"cuda:{cfg.device}" if cfg.use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.log.info(f"Using device {self.device}")

        # dtype
        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)
        if cfg.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

    @abstractmethod
    def run(self):
        """A function defining the execution of the experiment. To be implemented by the child class."""
        pass

    def log_resources(self):

        max_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.log.info(f"Peak system RAM usage: {max_ram/1024**2:.3} GB")

        if self.cfg.use_gpu and self.device.type == "cuda":
            max_mem_gpu = torch.cuda.max_memory_allocated(self.device)
            tot_mem_gpu = torch.cuda.mem_get_info(self.device)[1]
            GB = 1024**3
            self.log.info(
                f"Peak GPU RAM usage: {max_mem_gpu/GB:.3} GB (of {tot_mem_gpu/GB:.3} GB available)"
            )
