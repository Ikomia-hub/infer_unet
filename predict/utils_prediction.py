import PIL.Image
import torch
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
import numpy as np


def predict_mask(net, full_img, device, scale_factor=0.5, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # net output shape: (Batch_size, Num_classes, h, w)
        output = net(img)
        # output scores to probabilities
        probs = F.softmax(output, dim=1)[0]

        # transform prediction to PIL image with the same size as input image
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])
        # remove the added dim (batch_size) --> shape: (Num_classes, h, w)
        full_mask = tf(probs.cpu()).squeeze()

        if net.n_classes == 1:
            mask = (full_mask > out_threshold).numpy()
        else:
            mask = F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

        if mask.ndim == 2:
            return mask
        elif mask.ndim == 3:
            return np.argmax(mask, axis=0) # * 255 / mask.shape[0]


def preprocess(pil_img, scale, is_mask):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img_ndarray = np.asarray(pil_img)
    return img_ndarray.transpose((2, 0, 1))


def mask_to_image(mask: np.ndarray):
    return mask * 255