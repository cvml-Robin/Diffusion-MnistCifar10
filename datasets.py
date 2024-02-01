import torchvision
import torch.utils.data as data

# train_data = torchvision.datasets.MNIST(
#     download=False,
#     root='./dataset',
#     transform=torchvision.transforms.ToTensor(),
#     train=False
# )
train_data = torchvision.datasets.CIFAR10(
    download=True,
    root='./dataset',
    transform=torchvision.transforms.ToTensor(),
    train=True
)
train_loader = data.DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=12
)

for ii in range(1000):
    img, _ = next(iter(train_loader))
    img = torchvision.transforms.Grayscale()(img)
    print(img.shape)
