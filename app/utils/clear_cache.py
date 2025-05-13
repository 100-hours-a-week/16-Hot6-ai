import gc
import torch

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()