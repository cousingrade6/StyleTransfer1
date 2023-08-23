import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print(device)

# load image
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([transforms.Resize([imsize, imsize]),
                             transforms.ToTensor()])


def image_loader(image_name):
    # load image and transform image to tensor
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)  # add one dimension as batch_size
    return image.to(device, torch.float)    # don't forget to deliver values to gpu


basic_path = r'E:/ArtisticStyle'
style_img = image_loader(basic_path + r'/style/Vangogh.jpg')
content_img = image_loader(basic_path + r'/content.jpg')
print("style_img's device: ", style_img.device)

# show image
unloader = transforms.ToPILImage()
plt.ion()  # 开启交互


def imshow(imtensor, title=None):
    image = imtensor.cpu().clone()
    image = image.squeeze(0)  # back to 3 dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()
    plt.pause(0.001)


plt.figure()
imshow(style_img, title='Style Image')
plt.figure()
imshow(content_img, title='Content Image')


class ContentLoss(nn.Module):
    # compute content loss and add this content loss module directly after the convolution
    # layer(s) that are being used to compute the content distance.
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.loss = None
        # we detach the target content from the tree
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(self.target, x)
        return x


def get_gram(x):
    batch_size, num_channels, w, h = x.size()
    feature = x.view(batch_size * num_channels, w * h)  # 将4维张量转换为2为矩阵
    gram = torch.mm(feature, feature.t())
    return gram.div(batch_size * num_channels * w * h)


class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.loss = None
        self.target = get_gram(target.detach())

    def forward(self, x):
        gram = get_gram(x)
        self.loss = F.mse_loss(self.target, gram)
        return x


# import VGG net
vgg_net = models.vgg19(pretrained=True).features.eval()
print("net's device: ", next(vgg_net.parameters()).device)

# VGG are trained on images with each channel normalized by
norm_mean = torch.tensor([0.485, 0.456, 0.406])
norm_std = torch.tensor([0.229, 0.224, 0.225])
print("norm's device: ", norm_mean.device)


# normalization module
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # view the mean and std to 3 dimension
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std


# select following conv layers to compute loss
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# Insert ContentLoss and StyleLoss to model
def rebuild_model(net, style_image, content_image,
                  style_layers=None,
                  content_layers=None):
    if content_layers is None:
        content_layers = content_layers_default
    if style_layers is None:
        style_layers = style_layers_default
    net = copy.deepcopy(net)  # 深拷贝
    content_losses = []
    style_losses = []

    norm_layer = Normalization(norm_mean, norm_std)
    model = nn.Sequential(norm_layer)

    i = 0  # increment every time we meet a conv
    for layer in net.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
            name = 'relu_{}'.format(i)
        elif isinstance(layer, nn.MaxPool2d):
            # As the paper, replacing the
            # max-pooling operation by average pooling improves the gradient flow.
            # But I find not changing it makes the final work better
            # layer = nn.AvgPool2d(layer.kernel_size)
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'batch_norm_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # insert StyleLoss or ContentLoss after specific conv_layer
        if name in style_layers:
            target = model(style_image).detach()  # style image does not take part in training
            style_loss_layer = StyleLoss(target)
            model.add_module('style_layer_{}'.format(i), style_loss_layer)
            style_losses.append(style_loss_layer)

        if name in content_layers:
            target = model(content_image).detach()
            content_loss_layer = ContentLoss(target)
            model.add_module('content_layer_{}'.format(i), content_loss_layer)
            content_losses.append(content_loss_layer)

    # trim off redundant layers
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], StyleLoss) or isinstance(model[i], ContentLoss):
            model = model[:(i + 1)]  # discard layers after the last StyleLoss or ContentLoss
            break
    print(model)
    return model, style_losses, content_losses


# Select input image, noise figure or original figure
input_img = content_img.clone().to(device)
print("input image's device: ", input_img.device)


# train model and get the final work
def runStyleTransfer(net, style_image, content_image, input_image, num_steps=300,
                     style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = rebuild_model(net, style_image, content_image)

    # We want to optimize the input and not the model parameters, so we
    # update all the requires_grad fields accordingly
    input_image.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = torch.optim.LBFGS([input_image.requires_grad_()])

    print('----------training-----------')

    step = [0]
    while step[0] < num_steps:
        def closure():
            input_image.data.clamp_(0, 1)       # confine to (0, 1) after every step
            optimizer.zero_grad()
            model(input_image)
            style_score = 0.0
            content_score = 0.0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()

            step[0] += 1
            if step[0] % 50 == 0:
                print("run {}".format(step))
                print('Style Loss: {:4f} Content Loss: {:4f}'.format(
                    style_score, content_score))
                print()
            return content_score + style_score
        optimizer.step(closure)
    with torch.no_grad():
        input_image.clamp_(0, 1)
    return input_image


# start training
output = runStyleTransfer(vgg_net, style_img, content_img, input_img)
plt.figure()
imshow(output, title='Final Work')

plt.ioff()
plt.show()  # 图片输出




