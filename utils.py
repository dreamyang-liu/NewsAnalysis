import torch

def gpu_memory_collect():
    for i in range(20):
        torch.cuda.empty_cache()

def freeze_model(model):
    for param in model.parameters():
        param.data = param.data.to(torch.float16)
        param.requires_grad = False

if __name__ == '__main__':
    pass