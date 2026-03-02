"""
Interactive Latent Space Explorer
==================================
Load a trained VAE model and interactively explore the latent space.

Usage:
    python explore_latent_space.py

Features:
- Generate digits from specific latent coordinates
- Interpolate between two points in latent space
- Sample random points and see what they generate
- Explore digit transitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class VAE(nn.Module):
    """VAE architecture (must match training script)."""
    
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decode(z), mu, logvar


def load_model(model_path='vae_model.pth'):
    """Load a trained VAE model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    model = VAE(latent_dim=config['latent_dim'], hidden_dim=config['hidden_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Latent dimensions: {config['latent_dim']}")
    print(f"Training epochs: {config['epochs']}")
    print(f"Final train loss: {checkpoint['train_losses'][-1]:.2f}")
    print(f"Final test loss: {checkpoint['test_losses'][-1]:.2f}\n")
    
    return model


def generate_from_point(model, x, y, device='cpu'):
    """Generate a digit from a specific latent space point."""
    z = torch.tensor([[x, y]], dtype=torch.float32).to(device)
    with torch.no_grad():
        digit = model.decode(z)
    return digit.cpu().view(28, 28).numpy()


def interpolate(model, point1, point2, steps=10, device='cpu'):
    """
    Interpolate between two points in latent space.
    
    Args:
        point1: (x1, y1) starting point
        point2: (x2, y2) ending point
        steps: number of interpolation steps
    """
    x1, y1 = point1
    x2, y2 = point2
    
    # Linear interpolation
    alphas = np.linspace(0, 1, steps)
    images = []
    
    with torch.no_grad():
        for alpha in alphas:
            x = x1 * (1 - alpha) + x2 * alpha
            y = y1 * (1 - alpha) + y2 * alpha
            digit = generate_from_point(model, x, y, device)
            images.append(digit)
    
    # Plot interpolation
    fig, axes = plt.subplots(1, steps, figsize=(steps*1.5, 2))
    for i, (img, alpha) in enumerate(zip(images, alphas)):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'α={alpha:.1f}', fontsize=8)
    
    plt.suptitle(f'Interpolation from {point1} to {point2}')
    plt.tight_layout()
    plt.savefig('interpolation.png', dpi=150, bbox_inches='tight')
    print(f"Interpolation saved to interpolation.png")
    plt.show()


def explore_grid(model, x_range=(-3, 3), y_range=(-3, 3), grid_size=5, device='cpu'):
    """
    Generate a grid of digits from a region of latent space.
    
    Args:
        x_range: (min, max) for x coordinate
        y_range: (min, max) for y coordinate
        grid_size: number of samples in each dimension
    """
    x_values = np.linspace(x_range[0], x_range[1], grid_size)
    y_values = np.linspace(y_range[0], y_range[1], grid_size)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*1.5, grid_size*1.5))
    
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            digit = generate_from_point(model, x, y, device)
            axes[i, j].imshow(digit, cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'({x:.1f},{y:.1f})', fontsize=6)
    
    plt.suptitle(f'Grid Exploration: x∈{x_range}, y∈{y_range}')
    plt.tight_layout()
    plt.savefig('grid_exploration.png', dpi=150, bbox_inches='tight')
    print(f"Grid exploration saved to grid_exploration.png")
    plt.show()


def sample_random(model, n=10, scale=2.0, device='cpu'):
    """
    Sample random points from the latent space.
    
    Args:
        n: number of samples
        scale: standard deviation for sampling (default 2.0 covers most learned space)
    """
    fig, axes = plt.subplots(1, n, figsize=(n*1.5, 2))
    
    with torch.no_grad():
        # Sample from N(0, scale^2)
        z = torch.randn(n, 2) * scale
        
        for i in range(n):
            digit = model.decode(z[i:i+1]).cpu().view(28, 28).numpy()
            axes[i].imshow(digit, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'({z[i,0]:.1f},{z[i,1]:.1f})', fontsize=8)
    
    plt.suptitle(f'Random Samples from N(0, {scale}²)')
    plt.tight_layout()
    plt.savefig('random_samples.png', dpi=150, bbox_inches='tight')
    print(f"Random samples saved to random_samples.png")
    plt.show()


def circular_path(model, center=(0, 0), radius=2.0, steps=16, device='cpu'):
    """
    Generate digits along a circular path in latent space.
    Shows how digits change as you move in a circle.
    
    Args:
        center: (x, y) center of circle
        radius: radius of the circle
        steps: number of points on the circle
    """
    angles = np.linspace(0, 2*np.pi, steps, endpoint=False)
    images = []
    
    with torch.no_grad():
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            digit = generate_from_point(model, x, y, device)
            images.append(digit)
    
    # Plot circular path
    fig, axes = plt.subplots(2, steps//2, figsize=(steps, 4))
    axes = axes.flatten()
    
    for i, (img, angle) in enumerate(zip(images, angles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'{np.degrees(angle):.0f}°', fontsize=8)
    
    plt.suptitle(f'Circular Path: center={center}, radius={radius}')
    plt.tight_layout()
    plt.savefig('circular_path.png', dpi=150, bbox_inches='tight')
    print(f"Circular path saved to circular_path.png")
    plt.show()


def main():
    """Interactive exploration menu."""
    
    print("="*60)
    print("VAE Latent Space Explorer")
    print("="*60)
    print()
    
    # Load model
    try:
        model = load_model('vae_model.pth')
    except FileNotFoundError:
        print("Error: vae_model.pth not found!")
        print("Please train the model first using: python vae_mnist.py")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    while True:
        print("\nExploration Options:")
        print("1. Generate from specific point")
        print("2. Interpolate between two points")
        print("3. Explore grid region")
        print("4. Sample random points")
        print("5. Circular path")
        print("6. Run all demonstrations")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-6): ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
            
        elif choice == '1':
            x = float(input("Enter x coordinate: "))
            y = float(input("Enter y coordinate: "))
            digit = generate_from_point(model, x, y, device)
            plt.imshow(digit, cmap='gray')
            plt.title(f'Generated digit at ({x}, {y})')
            plt.axis('off')
            plt.savefig(f'point_{x}_{y}.png', dpi=150, bbox_inches='tight')
            print(f"Saved to point_{x}_{y}.png")
            plt.show()
            
        elif choice == '2':
            print("First point:")
            x1 = float(input("  x1: "))
            y1 = float(input("  y1: "))
            print("Second point:")
            x2 = float(input("  x2: "))
            y2 = float(input("  y2: "))
            steps = int(input("Number of steps (default 10): ") or "10")
            interpolate(model, (x1, y1), (x2, y2), steps, device)
            
        elif choice == '3':
            print("Grid region:")
            x_min = float(input("  x min (default -3): ") or "-3")
            x_max = float(input("  x max (default 3): ") or "3")
            y_min = float(input("  y min (default -3): ") or "-3")
            y_max = float(input("  y max (default 3): ") or "3")
            size = int(input("  grid size (default 5): ") or "5")
            explore_grid(model, (x_min, x_max), (y_min, y_max), size, device)
            
        elif choice == '4':
            n = int(input("Number of samples (default 10): ") or "10")
            scale = float(input("Sampling scale (default 2.0): ") or "2.0")
            sample_random(model, n, scale, device)
            
        elif choice == '5':
            print("Circular path:")
            cx = float(input("  center x (default 0): ") or "0")
            cy = float(input("  center y (default 0): ") or "0")
            r = float(input("  radius (default 2.0): ") or "2.0")
            steps = int(input("  steps (default 16): ") or "16")
            circular_path(model, (cx, cy), r, steps, device)
            
        elif choice == '6':
            print("\nRunning all demonstrations...\n")
            print("1. Interpolating between (-2, -2) and (2, 2)...")
            interpolate(model, (-2, -2), (2, 2), 10, device)
            
            print("\n2. Exploring grid from (-3, -3) to (3, 3)...")
            explore_grid(model, (-3, 3), (-3, 3), 5, device)
            
            print("\n3. Sampling 12 random points...")
            sample_random(model, 12, 2.0, device)
            
            print("\n4. Circular path around origin...")
            circular_path(model, (0, 0), 2.0, 16, device)
            
            print("\nAll demonstrations complete!")
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    main()