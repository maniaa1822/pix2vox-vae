import os
import torch
import numpy as np
from PIL import Image
from utils.binvox_visualization import get_volume_views
from test_net import test_net, make_grid_image
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from utils.network_utils import var_or_cuda

class Visualizer:
    def __init__(self, config):
        self.cfg = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.refiner = Refiner(config)
        self.merger = Merger(config)

        if torch.cuda.is_available():
            self.encoder = torch.nn.DataParallel(self.encoder).cuda()
            self.decoder = torch.nn.DataParallel(self.decoder).cuda()
            self.refiner = torch.nn.DataParallel(self.refiner).cuda()
            self.merger = torch.nn.DataParallel(self.merger).cuda()

        print('[INFO] Loading weights from %s ...' % config.CONST.WEIGHTS)
        checkpoint = torch.load(config.CONST.WEIGHTS)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if config.NETWORK.USE_REFINER:
            self.refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if config.NETWORK.USE_MERGER:
            self.merger.load_state_dict(checkpoint['merger_state_dict'])

        self.encoder.eval()
        self.decoder.eval()
        self.refiner.eval()
        self.merger.eval()

    def visualize(self, images, output_dir):
        with torch.no_grad():
            images = var_or_cuda(images)
            mu, log_sigma, z = self.encoder(images)
            raw_features, generated_volume = self.decoder(z)

            if self.cfg.NETWORK.USE_MERGER and self.cfg.TRAIN.EPOCH_START_USE_MERGER >= 0:
                generated_volume = self.merger(raw_features, generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)

            if self.cfg.NETWORK.USE_REFINER and self.cfg.TRAIN.EPOCH_START_USE_REFINER >= 0:
                generated_volume = self.refiner(generated_volume)

            gv = generated_volume.cpu().numpy()
            rendering_views = get_volume_views(gv, os.path.join(output_dir, 'test'), -1)
            grid_image = make_grid_image(rendering_views)
            Image.fromarray(grid_image).save(os.path.join(output_dir, 'visualization.png'))

            return generated_volume

# Usage example
if __name__ == "__main__":
    from utils.config import cfg
    visualizer = Visualizer(cfg)

    # Load input images
    input_images = [np.array(Image.open('input_image1.png')),
                    np.array(Image.open('input_image2.png')),
                    np.array(Image.open('input_image3.png'))]
    input_images = torch.Tensor(input_images)

    # Visualize the 3D result
    output_dir = 'path/to/output/directory'
    os.makedirs(output_dir, exist_ok=True)
    generated_volume = visualizer.visualize(input_images, output_dir)
    print('Visualization saved to:', os.path.join(output_dir, 'visualization.png'))