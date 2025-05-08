import torch

def initialize_cuda():
    if torch.cuda.is_available():
        _ = torch.tensor([0.0], device="cuda")
        torch.cuda.synchronize()
        print(f"[INFO] CUDA initialized: {torch.cuda.get_device_name(0)}")