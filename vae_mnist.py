"""
Variational Autoencoder (VAE) for MNIST
========================================
A complete implementation of VAE with latent space exploration capabilities.

Key Concepts:
- Encoder: Maps input to latent distribution (mean and log-variance)
- Decoder: Reconstructs input from latent samples
- Reparameterization trick: Enables backpropagation through sampling
- Loss: Reconstruction loss + KL divergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class VAE(nn.Module):
    """Variational Autoencoder with configurable latent dimensions."""
    
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        """
        Args:
            input_dim: Size of flattened input (28*28=784 for MNIST)
            hidden_dim: Size of hidden layers
            latent_dim: Dimensionality of latent space (2 for easy visualization)
        """
        super(VAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean of latent distribution
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance of latent distribution
        
        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        """
        Encode input to latent distribution parameters.
        Returns: mean and log-variance of latent distribution
        """
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        where epsilon ~ N(0,1)
        
        This allows gradients to flow through the sampling process.
        """
        std = torch.exp(0.5 * logvar)  # Convert log-variance to std
        eps = torch.randn_like(std)  # Sample from standard normal
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent vector to reconstructed input."""
        h = F.relu(self.fc3(z))
        reconstruction = torch.sigmoid(self.fc4(h))
        return reconstruction
    
    def forward(self, x):
        """
        Forward pass through VAE.
        Returns: reconstruction, mean, log-variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def vae_loss(reconstruction, x, mu, logvar):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Reconstruction loss: Binary cross-entropy (how well we reconstruct)
    KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0,1)
                   Regularizes the latent space to be close to standard normal
    
    The KL divergence has a closed form:
    KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    # Reconstruction loss (binary cross-entropy)
    BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    
    # KL divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)  # Flatten images
        optimizer.zero_grad()
        
        # Forward pass
        reconstruction, mu, logvar = model(data)
        
        # Compute loss
        loss = vae_loss(reconstruction, data, mu, logvar)
        
        # Backward pass
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    return train_loss / len(train_loader.dataset)


def test_epoch(model, test_loader, device):
    """Evaluate on test set."""
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, 784).to(device)
            reconstruction, mu, logvar = model(data)
            test_loss += vae_loss(reconstruction, data, mu, logvar).item()
    
    return test_loss / len(test_loader.dataset)


def visualize_latent_space(model, test_loader, device, save_path='latent_space.png'):
    """
    Visualize the 2D latent space colored by digit class.
    Only works with latent_dim=2.
    """
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.view(-1, 784).to(device)
            mu, _ = model.encode(data)
            latent_vectors.append(mu.cpu().numpy())
            labels.append(label.numpy())
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=5)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('MNIST Latent Space Representation')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Latent space visualization saved to {save_path}")
    plt.close()


def visualize_manifold(model, device, n=20, digit_size=28, save_path='manifold.png'):
    """
    Generate a grid of digits by sampling the latent space.
    Creates a 2D manifold visualization for latent_dim=2.
    """
    model.eval()
    
    # Create a grid of latent values
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)
    
    figure = np.zeros((digit_size * n, digit_size * n))
    
    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                reconstruction = model.decode(z)
                digit = reconstruction.cpu().view(digit_size, digit_size).numpy()
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
    
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.axis('off')
    plt.title('Latent Space Manifold (Sampling Grid)')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Manifold visualization saved to {save_path}")
    plt.close()


def visualize_reconstructions(model, test_loader, device, n=10, save_path='reconstructions.png'):
    """Compare original images with their reconstructions."""
    model.eval()
    
    # Get a batch of test images
    data, _ = next(iter(test_loader))
    data = data[:n].to(device)
    
    with torch.no_grad():
        data_flat = data.view(-1, 784)
        reconstruction, _, _ = model(data_flat)
        reconstruction = reconstruction.view(-1, 1, 28, 28)
    
    # Plot original and reconstructed
    fig, axes = plt.subplots(2, n, figsize=(n*1.5, 3))
    
    for i in range(n):
        # Original
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstruction
        axes[1, i].imshow(reconstruction[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Reconstruction comparison saved to {save_path}")
    plt.close()


def main():
    """Main training and visualization pipeline."""
    
    # Configuration
    config = {
        'batch_size': 128,
        'latent_dim': 2,  # 2D for visualization
        'hidden_dim': 400,
        'learning_rate': 1e-3,
        'epochs': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    device = torch.device(config['device'])
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    model = VAE(latent_dim=config['latent_dim'], hidden_dim=config['hidden_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"Model architecture:\n{model}\n")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Training loop
    print("Starting training...")
    train_losses = []
    test_losses = []
    
    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = test_epoch(model, test_loader, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f'Epoch {epoch:2d}/{config["epochs"]}: '
              f'Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')
    
    print("\nTraining complete!\n")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to training_curves.png")
    plt.close()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_reconstructions(model, test_loader, device)
    visualize_latent_space(model, test_loader, device)
    visualize_manifold(model, device)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'test_losses': test_losses
    }, 'vae_model.pth')
    print("\nModel saved to vae_model.pth")
    
    print("\n" + "="*60)
    print("VAE Training and Visualization Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - vae_model.pth: Trained model weights")
    print("  - training_curves.png: Loss curves over training")
    print("  - reconstructions.png: Original vs reconstructed images")
    print("  - latent_space.png: 2D latent space colored by digit")
    print("  - manifold.png: Grid of generated digits from latent space")
    print("\nLatent Space Exploration:")
    print("  The latent space visualization shows how different digits")
    print("  cluster in the 2D space. Similar digits are close together.")
    print("  The manifold shows smooth transitions between digit types.")


if __name__ == '__main__':
    main()