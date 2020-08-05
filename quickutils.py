import torch

def to_tensor(x):
    return torch.from_numpy(x).type(torch.float32).cuda()

class Logger():
    def __init__(self, **kwargs):
        self.dict = kwargs
        
    def peek(self, arg):
        return self.dict[arg][-1]
    
    def put(self, arg, val):
        return self.dict[arg].append(val)
    
    def summarize(self, iteration):
        print(f"Iter {iteration}, " + "".join(f"{key}: {value[-1]:.4f}, " for key, value in self.dict.items()))