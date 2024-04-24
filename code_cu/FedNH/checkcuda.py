import torch

# Kiểm tra số lượng GPU CUDA
num_cuda_devices = torch.cuda.device_count()
print("Số lượng GPU CUDA:", num_cuda_devices)
