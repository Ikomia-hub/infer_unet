import PIL.Image
import torch
import torch.nn.functional as F
import os
from torchvision import transforms
from PIL import Image
import numpy as np


def predict_mask(net, full_img, device, size, out_threshold=0.5):
    print('full_img', (np.array(full_img)).shape)
    net.eval()
    img = torch.from_numpy(preprocess(full_img, size, is_mask=False))
    print('img', img.shape)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # net output shape: (Batch_size, Num_classes, h, w)
        output = net(img)
        print('outp shape', output.shape)
        # output scores to probabilities
        probs = F.softmax(output, dim=1)[0]

        print('probs shape', probs.shape)

        # remove the added dim (batch_size) --> shape: (Num_classes, h, w)
        full_mask = probs.cpu().squeeze()

        if net.n_classes == 1:
            mask = (full_mask > out_threshold).numpy()
        else:
            mask = F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

        if mask.ndim == 2:
            return mask
        elif mask.ndim == 3:
            return np.argmax(mask, axis=0) # * 255 / mask.shape[0]


def preprocess(pil_img, size_h_w, is_mask):
    newW = size_h_w
    newH = size_h_w
    assert newW > 0 and newH > 0, 'img size is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img_ndarray = np.asarray(pil_img)
    return img_ndarray.transpose((2, 0, 1))


def mask_to_image(mask: np.ndarray):
    return mask * 255