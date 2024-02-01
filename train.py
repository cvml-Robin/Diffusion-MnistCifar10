import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import numpy as np
from torchvision.utils import save_image
from model import UNet, DDPM
from visdom import Visdom

train_vis = Visdom(env='Diffusion MNIST')
train_vis.close()
gpu = torch.device("cuda")
cpu = torch.device("cpu")
# train_data = torchvision.datasets.MNIST(
#     download=False,
#     root='./dataset',
#     transform=torchvision.transforms.ToTensor(),
#     train=False
# )

train_data = torchvision.datasets.CIFAR10(
    download=False,
    root='./dataset',
    transform=torchvision.transforms.ToTensor(),
    train=True
)
batch_size = 16
train_loader = data.DataLoader(
    dataset=train_data,
    shuffle=True,
    drop_last=True,
    batch_size=batch_size
)
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
gen_optim = torch.optim.Adam(gen_model.parameters(), lr=5e-5, betas=(0.5, 0.999))

max_iter = 1000000
loss_fcn = nn.MSELoss(reduction='mean').to(gpu)
print_freq = 1000
save_freq = 50000
print_loss = 0
for now_step in range(1, max_iter + 1):
    img, _ = next(iter(train_loader))
    img = torchvision.transforms.Grayscale()(img)
    img = img.to(gpu)
    time_label = torch.randint(0, time_n_steps, (batch_size,)).to(gpu)
    eps = torch.randn_like(img).to(gpu)
    x_t = ddpm.sample_forward(img, time_label, eps)
    eps_theta = gen_model(x_t, time_label.reshape(batch_size, 1))
    gen_optim.zero_grad()
    loss = loss_fcn(eps_theta, eps)
    loss.backward()
    gen_optim.step()

    if now_step % print_freq == 0:
        gen_model.eval()
        print(print_loss)
        train_vis.line(
            X=[now_step],
            Y=[print_loss],
            opts=dict(
                title='Loss',
                linecolor=np.array([[255, 0, 0]])
            ),
            update='append',
            win='Loss'
        )

        sample_img = ddpm.sample_backward(
            device=gpu,
            simple_var=True,
            img_shape=(4, 1, 32, 32),
            net=gen_model
        )
        sample_img = sample_img.to(cpu).clamp(0, 1)

        save_image(
            sample_img,
            './image/train/%.8d.jpg' % now_step,
            nrow=2
        )
        train_vis.images(
            tensor=sample_img,
            nrow=2,
            win='Image'
        )

        print_loss = 0
        gen_model.train()
    else:
        print_loss += loss.item()/print_freq

    if now_step % save_freq == 0:
        torch.save(
            obj=gen_model.state_dict(),
            f='./check_point/model/model_%.8d.pth' % now_step
        )


