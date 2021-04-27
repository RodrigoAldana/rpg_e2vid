
import time
import math
from model.model import *
from model.submodules import *
from ttictoc import tic,toc
import numpy as np

net_conf64 = {'num_bins': 5,
            'recurrent_block_type': 'convlstm',
            'base_num_channels': 64,
            'num_encoders': 3,
            'num_residual_blocks': 2,
            'use_upsample_conv': False,
            'norm': 'BN'}

net_conf32 = {'num_bins': 5,
            'recurrent_block_type': 'convlstm',
            'base_num_channels': 32,
            'num_encoders': 3,
            'num_residual_blocks': 2,
            'use_upsample_conv': False,
            'norm': 'BN'}

net_conf16 = {'num_bins': 5,
            'recurrent_block_type': 'convlstm',
            'base_num_channels': 16,
            'num_encoders': 3,
            'num_residual_blocks': 2,
            'use_upsample_conv': False,
            'norm': 'BN'}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model64 = E2VIDRecurrent(net_conf64)
model64 = model64.to(device)

model32 = E2VIDRecurrent(net_conf32)
model32 = model32.to(device)

model16 = E2VIDRecurrent(net_conf16)
model16 = model16.to(device)

model = dyn_E2VIDRecurrent(net_conf32)
model = model.to(device)


input = torch.zeros((1,5,184,240))
input = input.to(device)

timing = np.array([])
for i in range(0,100):
    tic()
    output = model64(input,None)
    timing = np.append(timing,toc())
print("64 timing: ", np.mean(timing))


timing = np.array([])
for i in range(0,100):
    tic()
    output = model32(input,None)
    timing = np.append(timing,toc())
print("32 timing: ", np.mean(timing))

timing = np.array([])
for i in range(0,100):
    tic()
    output = model16(input,None)
    timing = np.append(timing,toc())
print("16 timing: ", np.mean(timing))


timing = np.array([])
for i in range(0,100):
    tic()
    output = model(input,None)
    timing = np.append(timing,toc())
print("mode 0 timing: ", np.mean(timing))

model.set_latency_mode(1)
timing = np.array([])
for i in range(0,100):
    tic()
    output = model(input,None)
    timing = np.append(timing,toc())
print("mode 1 timing: ", np.mean(timing))