import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm

from config import cfg
from models.encoder import Encoder
import utils.data_loaders
import utils.data_transforms

def visualize_latent_space_2d(cfg, encoder, test_data_loader, taxonomies):
    encoder.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for taxonomy_id, sample_name, rendering_images, _ in tqdm(test_data_loader, desc="Processing samples"):
            rendering_images = rendering_images.cuda()
            mu, _, _ = encoder(rendering_images)
            latent_vectors.append(mu.cpu().numpy())
            labels.extend([taxonomies[tid]['taxonomy_name'] for tid in taxonomy_id])

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    print(f"Number of latent vectors: {len(latent_vectors)}")
    print(f"Number of labels: {len(labels)}")
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)

    # Plot
    plt.figure(figsize=(12, 8))
    unique_labels = list(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        mask = np.array(labels) == label
        plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], c=[color], label=label, alpha=0.7)
    
    plt.legend()
    plt.title("VAE Latent Space Visualization")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.savefig("latent_space_visualization.png")
    plt.close()
    
def vizualize_latent_space_3d(cfg, encoder, test_data_loader, taxonomies):
    encoder.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for taxonomy_id, sample_name, rendering_images, _ in tqdm(test_data_loader, desc="Processing samples"):
            rendering_images = rendering_images.cuda()
            mu, _, _ = encoder(rendering_images)
            latent_vectors.append(mu.cpu().numpy())
            labels.extend([taxonomies[tid]['taxonomy_name'] for tid in taxonomy_id])

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    
    print(f"Number of latent vectors: {len(latent_vectors)}")
    print(f"Number of labels: {len(labels)}")
    
    # Perform t-SNE
    tsne = TSNE(n_components=3, random_state=42)
    latent_3d = tsne.fit_transform(latent_vectors)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = list(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        mask = np.array(labels) == label
        ax.scatter(latent_3d[mask, 0], latent_3d[mask, 1], latent_3d[mask, 2], c=[color], label=label, alpha=0.7)
    
    ax.legend()
    ax.set_title("VAE Latent Space Visualization")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_zlabel("t-SNE dimension 3")
    plt.tight_layout()
    plt.savefig("latent_space_visualization_3d.png")
    plt.close()

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
    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()

    # Load pre-trained weights
    print('Loading weights from %s ...' % cfg.CONST.WEIGHTS)
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    # Visualize latent space
    visualize_latent_space_2d(cfg, encoder, test_data_loader, taxonomies)
    vizualize_latent_space_3d(cfg, encoder, test_data_loader, taxonomies)

if __name__ == '__main__':
    main()