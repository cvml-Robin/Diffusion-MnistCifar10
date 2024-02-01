import torch
import torch.nn as nn
import numpy as np
from model import UNet, DDPM
from torchvision.utils import save_image
import matplotlib.pyplot as plt

gpu = torch.device("cuda")
cpu = torch.device("cpu")
time_n_steps = 1000
ddpm = DDPM(
    device=gpu,
    n_steps=time_n_steps,
    min_beta=0.0001,
    max_beta=0.02)

gen_model = UNet(n_steps=time_n_steps,
                 channels=[10, 20, 40, 80],
                 pe_dim=128,
                 residual=True,
                 input_img_shape=[1, 32, 32]).to(gpu)
gen_model.load_state_dict(torch.load('./check_point/model/model_01000000.pth'))
gen_model.eval()
sample_times = 40
# save_img = torch.zeros((1, 1, 32 * sample_times, 32 * sample_times))
save_img = np.zeros((32 * sample_times, 32 * sample_times))
for ii in range(sample_times**2):
    sample_img = ddpm.sample_backward(
        device=gpu,
        net=gen_model,
        img_shape=(1, 1, 32, 32),
        simple_var=True
    ).to(cpu).clamp(0, 1).detach().data.numpy()[0, 0]
    row_id = int(ii / sample_times)
    col_id = (ii % sample_times)
    sx = col_id * 32
    sy = row_id * 32
    save_img[sy:(sy+32), sx:(sx+32)] = sample_img
# save_idx = 0
# while True:
#     save_idx += 1
#     sample_img = ddpm.sample_backward(
#         device=gpu,
#         net=gen_model,
#         img_shape=(1, 1, 32, 32),
#         simple_var=True
#     ).to(cpu).clamp(0, 1).data.numpy()[0, 0]
#     plt.imsave('./image/test/%d.png' % save_idx, sample_img)

save_img = np.dstack((save_img, save_img, save_img))
plt.imsave('./image/test/test_image_numpy.png', save_img)
