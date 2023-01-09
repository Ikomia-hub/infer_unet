import PIL.Image
import torch
import torch.nn.functional as F
from infer_unet.unet import UNet
from torchvision import transforms
from PIL import Image
import numpy as np


def predict_mask(net, full_img, device, size, out_threshold=0.5):
    net.eval()
    full_img = Image.fromarray(full_img)
    img = torch.from_numpy(preprocess(full_img, size, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # net output shape: (Batch_size, Num_classes, h, w)
        output = net(img)
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        # output scores to probabilities
        probs = F.softmax(output, dim=1)[0]

        # remove the added dim (batch_size) --> shape: (Num_classes, h, w)
        full_mask = probs.cpu().squeeze()

        if net.n_classes == 1:
            mask = (full_mask > out_threshold).numpy()
        else:
            mask = F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

        if mask.ndim == 2:
            return mask
        elif mask.ndim == 3:
            return np.argmax(mask, axis=0)  # * 255 / mask.shape[0]


def preprocess(pil_img, size_h_w, is_mask):
    assert size_h_w > 0, 'img size is too small, resized images would have no pixel'
    pil_img = pil_img.resize((size_h_w, size_h_w), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img_ndarray = np.asarray(pil_img)
    return img_ndarray.transpose((2, 0, 1))


def mask_to_image(mask: np.ndarray):
    return mask * 255


def unet_carvana(pretrained=True, scale=0.5):
    """
    UNet model trained on the Carvana dataset ( https://www.kaggle.com/c/carvana-image-masking-challenge/data ).
    Set the scale to 0.5 (50%) when predicting.
    """
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    if pretrained:
        if scale == 0.5:
            checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth'
        elif scale == 1.0:
            checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth'
        else:
            raise RuntimeError('Only 0.5 and 1.0 scales are available')
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True)
        if 'mask_values' in state_dict:
            state_dict.pop('mask_values')
        net.load_state_dict(state_dict)

    return net