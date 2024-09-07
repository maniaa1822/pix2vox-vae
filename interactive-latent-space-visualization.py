import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm
import matplotlib.backends.backend_agg as agg
import pygame
from pygame.locals import *

from config import cfg
from models.encoder import Encoder
from models.decoder import Decoder
from models.merger import Merger
from models.refiner import Refiner
import utils.data_loaders
import utils.data_transforms
import utils.binvox_visualization

def visualize_latent_space_interactive(cfg, encoder, decoder,merger,refiner, test_data_loader, taxonomies):
    encoder.eval()
    decoder.eval()
    merger.eval()
    refiner.eval()
    
    latent_vectors = []
    labels = []
    save_dir = '/home/matteo/AI_and_Robotics/CV/pix2vox-gen/visualization'
    with torch.no_grad():
        for taxonomy_id, sample_name, rendering_images, _ in tqdm(test_data_loader, desc="Processing samples"):
            rendering_images = rendering_images.cuda()
            mu, _, _ = encoder(rendering_images)
            latent_vectors.append(mu.cpu().numpy())
            labels.extend([taxonomies[tid]['taxonomy_name'] for tid in taxonomy_id])

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)

    # Create a mapping of unique labels to integers
    unique_labels = list(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    color_indices = [label_to_int[label] for label in labels]

    # Initialize Pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Interactive Latent Space Visualization")

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=color_indices, cmap='viridis', alpha=0.7)
    ax.set_title("VAE Latent Space Visualization")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")

    # Add a colorbar legend
    cbar = plt.colorbar(scatter)
    cbar.set_ticks(range(len(unique_labels)))
    cbar.set_ticklabels(unique_labels)

    # Convert Matplotlib figure to Pygame surface
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    surf = pygame.image.fromstring(raw_data, size, "RGB")

    running = True
    while running:
        screen.fill((255, 255, 255))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                x, y = event.pos
                # Convert screen coordinates to data coordinates
                inv = ax.transData.inverted()
                data_coords = inv.transform((x, y))
                
                # Find the nearest point in latent_2d
                distances = np.sum((latent_2d - data_coords)**2, axis=1)
                nearest_idx = np.argmin(distances)
                
                # Get the corresponding latent vector
                selected_latent_vector = torch.from_numpy(latent_vectors[nearest_idx]).unsqueeze(0).cuda()
                
                # Generate 3D model from the latent vector
                with torch.no_grad():
                    raw_features, generated_volume = decoder(selected_latent_vector)
                    generated_volume = merger(raw_features, generated_volume)
                    generated_volume = refiner(generated_volume)
                    #generated_volume = torch.mean(generated_volume, dim=1)

                # Visualize the generated 3D model
                gv = generated_volume.cpu().numpy()
                rendering_views = utils.binvox_visualization.get_volume_views_cv2(gv, save_dir, 0)
                
                # Display the generated 3D model
                plt.figure()
                plt.imshow(rendering_views)
                plt.title(f"Generated 3D Model (Label: {labels[nearest_idx]})")
                plt.axis('off')
                plt.show()

    pygame.quit()

def main():
    # Load taxonomies
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data transforms
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
        shuffle=False
    )

    # Set up network
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    merger = Merger(cfg)
    refiner = Refiner(cfg)
    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        merger = torch.nn.DataParallel(merger).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()

    # Load pre-trained weights
    print('Loading weights from %s ...' % cfg.CONST.WEIGHTS)
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Visualize latent space
    visualize_latent_space_interactive(cfg, encoder, decoder,merger,refiner, test_data_loader, taxonomies)

if __name__ == '__main__':
    main()