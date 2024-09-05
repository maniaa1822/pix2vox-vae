from datetime import datetime
import os
from PIL import Image
from typing import OrderedDict
 
import cv2
import numpy as np
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
 
from config import cfg
 
import torch
 
import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

encoder = Encoder(cfg)
decoder = Decoder(cfg)
refiner = Refiner(cfg)
merger = Merger(cfg)

cfg.CONST.WEIGHTS = '/kaggle/input/model/pytorch/default/1/best-ckpt.pth'
checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location=torch.device('cpu'))

fix_checkpoint = {}
fix_checkpoint['encoder_state_dict'] = OrderedDict((k.split('module.')[1:][0], v) for k, v in checkpoint['encoder_state_dict'].items())
fix_checkpoint['decoder_state_dict'] = OrderedDict((k.split('module.')[1:][0], v) for k, v in checkpoint['decoder_state_dict'].items())

epoch_idx = checkpoint['epoch_idx']
encoder.load_state_dict(fix_checkpoint['encoder_state_dict'])
decoder.load_state_dict(fix_checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()
refiner.eval()
merger.eval()

img1_path = '/kaggle/input/shapenet/ShapeNetRendering/ShapeNetRendering/02691156/10155655850468db78d106ce0a280f87/rendering/00.png'
img1_np = cv2.imread(img1_path)

sample = np.array([img1_np])

IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W

test_transforms = utils.data_transforms.Compose([
    utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
    utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE), 
    utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
    utils.data_transforms.ToTensor(),
])

rendering_images = test_transforms(rendering_images=sample)
rendering_images = rendering_images.unsqueeze(0)

with torch.no_grad():
    image_features = encoder(rendering_images)
    #convert image feature to torch tensor
    # Test the encoder, decoder, refiner and merger
    mu, log_sigma, z = encoder(rendering_images)
    raw_features, generated_volume = decoder(z)
    
    if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
        generated_volume = merger(raw_features, generated_volume)
    else:
        generated_volume = torch.mean(generated_volume, dim=1)


generated_volume = generated_volume.squeeze(0)

img_dir= 'output_images'
gv = generated_volume.cpu().numpy()
rendering_views = utils.binvox_visualization.get_volume_views_cv2(gv, os.path.join(img_dir, 'test'),epoch_idx)
rendering_views = np.swapaxes(rendering_views, 2, 0)
