import torch
import sys
print(sys.version)
is_available:bool = torch.cuda.is_available()
if(is_available):
    print(torch.cuda.get_device_name(0))
else:
    print("GPU:",is_available)