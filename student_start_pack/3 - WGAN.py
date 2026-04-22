# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: genai-env
#     language: python
#     name: python3
# ---

# %% [markdown]
# Imports

# %%
import sys
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import utils as vutils
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import optuna
from torchvision import transforms as T

if "__file__" in globals():
    FILE_DIR = Path(__file__).resolve().parent
else:
    FILE_DIR = Path.cwd()

PROJECT_ROOT = FILE_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
KAGGLE_ROOT = PROJECT_ROOT / 'ArtBench-10'

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artbench_local_dataset import load_kaggle_artbench10_splits


# %% [markdown]
# Utils

# %%
class HFDatasetTorch(Dataset):
    def __init__(self, hf_split, transform=None, indices=None):
        self.ds = hf_split
        self.transform = transform
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        ex = self.ds[real_idx]
        img = ex["image"]
        y = int(ex["label"])
        x = self.transform(img) if self.transform else img
        return x, y

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')

device = get_device()

def denorm(x):
    return (x * 0.5) + 0.5

def show_image_grid(images, title='Generated Images', n_show=25, save_path=None):
    images = denorm(images.detach().cpu())
    grid = vutils.make_grid(images[:n_show], nrow=int(np.sqrt(n_show)), normalize=False)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).clamp(0, 1))
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
print(device)

# %% [markdown]
# DATA LOADERS 

# %%
# Cache the dataset globally to avoid reloading from disk in every Optuna trial
print("Loading ArtBench-10 dataset")
GLOBAL_HF_DS = load_kaggle_artbench10_splits(KAGGLE_ROOT)

def build_loaders(batch_size=128, train_limit=None, test_limit=None):
    transform = T.Compose([
        T.Resize(32),
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_ds = HFDatasetTorch(GLOBAL_HF_DS["train"], transform=transform)
    test_ds = HFDatasetTorch(GLOBAL_HF_DS["test"], transform=transform)
    
    if train_limit: train_ds = Subset(train_ds, range(min(train_limit, len(train_ds))))
    if test_limit: test_ds = Subset(test_ds, range(min(test_limit, len(test_ds))))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader



# %% [markdown]
# WGAN implementation

# %%
class WGANGPGenerator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64, channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))

class WGANGPCritic(nn.Module):
    def __init__(self, ndf=64, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.LayerNorm([ndf * 2, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LayerNorm([ndf * 4, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



# %%
def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones(real_samples.size(0), 1, device=device, requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



# %% [markdown]
# EVALUATION
#

# %%
def evaluate_metrics(generator, dataloader, latent_dim, device, num_samples=5000):
    generator.eval()
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=100, subsets=50, normalize=True).to(device)

    # Real images
    count = 0
    for real_imgs, _ in dataloader:
        if count >= num_samples: break
        batch = real_imgs[:num_samples-count].to(device)
        batch_01 = denorm(batch).clamp(0, 1)
        fid.update(batch_01, real=True)
        kid.update(batch_01, real=True)
        count += batch.size(0)

    # Generated images
    count = 0
    while count < num_samples:
        batch_size = min(100, num_samples - count)
        z = torch.randn(batch_size, latent_dim, device=device)
        with torch.no_grad():
            fake_imgs = generator(z)
        fake_01 = denorm(fake_imgs).clamp(0, 1)
        fid.update(fake_01, real=False)
        kid.update(fake_01, real=False)
        count += batch_size

    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    return fid_score, kid_mean.item(), kid_std.item()

def run_robust_evaluation(generator, dataloader, latent_dim, device, num_runs=10):
    fids, kids = [], []
    print(f"Starting robust evaluation ({num_runs} runs)...")
    for i in range(num_runs):
        set_seed(100 + i)
        f, k, _ = evaluate_metrics(generator, dataloader, latent_dim, device)
        fids.append(f)
        kids.append(k)
        print(f"Run {i+1}: FID={f:.4f}, KID={k:.4f}")
    print(f"\nFinal FID: {np.mean(fids):.4f} ± {np.std(fids):.4f}")
    print(f"Final KID: {np.mean(kids):.4f} ± {np.std(kids):.4f}")



def save_wgan_checkpoint(generator, critic, history, checkpoint_path, params):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'gen_state': generator.state_dict(),
        'cri_state': critic.state_dict(),
        'params': params,
        'history': history
    }, checkpoint_path)
    print(f"✅ Checkpoint saved to {checkpoint_path}")

# %% [markdown]
# TRAINING
#

# %%
def train_wgan_gp(generator, critic, loader, latent_dim, epochs, lr, n_critic=5, lambda_gp=10, checkpoint_dir=None, val_loader=None, save_interval=10, num_fid_samples=1500, model_params=None, print_progress=True):
    # Stabilized Adam for WGAN-GP: beta1=0.0, beta2=0.99
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.99))
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.99))
    
    history = {'g_loss': [], 'c_loss': [], 'fid': []}
    best_fid = float('inf')
    
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / 'samples').mkdir(exist_ok=True)

    for epoch in range(epochs):
        g_loss_run, c_loss_run = 0.0, 0.0
        n_batches = 0
        
        generator.train()
        critic.train()

        for real, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, disable=not print_progress):
            real = real.to(device)
            bs = real.size(0)
            
            # Update Critic (n_critic updates per 1 generator update)
            for _ in range(n_critic):
                z = torch.randn(bs, latent_dim, device=device)
                fake = generator(z).detach()
                
                opt_c.zero_grad()
                c_real = critic(real)
                c_fake = critic(fake)
                gp = compute_gradient_penalty(critic, real, fake)
                
                # Wasserstein Loss + Gradient Penalty
                c_loss = torch.mean(c_fake) - torch.mean(c_real) + lambda_gp * gp
                
                # Adds a tiny penalty to keep the critic's output scores from drifting to infinity
                epsilon = 0.001
                drift_penalty = epsilon * (torch.mean(c_real ** 2))
                c_loss += drift_penalty
                c_loss.backward()
                opt_c.step()
                c_loss_run += c_loss.item()
            
            # Update Generator (using fresh noise)
            opt_g.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            fake = generator(z)
            g_loss = -torch.mean(critic(fake))
            g_loss.backward()
            opt_g.step()
            g_loss_run += g_loss.item()
            n_batches += 1
            
        avg_g = g_loss_run / n_batches
        avg_c = c_loss_run / (n_batches * n_critic)
        history['g_loss'].append(avg_g)
        history['c_loss'].append(avg_c)
        
        status_str = f"Epoch {epoch+1:02d}/{epochs} | G Loss: {avg_g:.4f} | C Loss: {avg_c:.4f}"

        # Periodic Monitoring: Visual Samples, FID, and Checkpoints
        if (epoch + 1) % save_interval == 0 or epoch == 0 or epoch == epochs - 1:
            # 1. Visual Samples
            if checkpoint_dir:
                with torch.no_grad():
                    generator.eval()
                    sample_noise = torch.randn(25, latent_dim, device=device)
                    samples = generator(sample_noise)
                    sample_path = checkpoint_dir / 'samples' / f'epoch_{epoch+1:03d}.png'
                    show_image_grid(samples, title=f'Epoch {epoch+1}', save_path=sample_path)
                    generator.train()

            # 2. FID Metrics
            if val_loader:
                fid_score, kid_mean, _ = evaluate_metrics(generator, val_loader, latent_dim, device, num_samples=num_fid_samples)
                history['fid'].append({'epoch': epoch + 1, 'fid': fid_score, 'kid': kid_mean})
                status_str += f" | FID: {fid_score:.2f}"
                
                # Save best model based on FID
                if fid_score < best_fid and checkpoint_dir:
                    best_fid = fid_score
                    save_wgan_checkpoint(generator, critic, history, checkpoint_dir / 'best_fid_model.pt', model_params)
                    if print_progress:
                        print(f"   -> New best FID: {best_fid:.4f}. Saved best_fid_model.pt")

            # 3. Regular Checkpoint
            if checkpoint_dir:
                save_wgan_checkpoint(generator, critic, history, checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pt', model_params)

        if print_progress:
            print(status_str)

    return history



# %% [markdown]
# OPTUNA optimization

# %%

def objective(trial):
    # Focused tuning: Learning Rate
    latent_dim = 128
    ngf = 64
    lr = trial.suggest_float('lr', 1e-4, 4e-4, log=True)
    
    print(f"\n>>> Trial {trial.number} | LR: {lr:.6f}")
    
    # Fast evaluation subset
    train_loader, val_loader = build_loaders(batch_size=128, train_limit=10000)
    
    gen = WGANGPGenerator(latent_dim, ngf).to(device)
    cri = WGANGPCritic(ngf).to(device)
    gen.apply(weights_init)
    cri.apply(weights_init)
    
    # n_critic=5, lambda_gp=10, beta1=0.0, beta2=0.99
    train_wgan_gp(gen, cri, train_loader, latent_dim, epochs=10, lr=lr)
    fid, _, _ = evaluate_metrics(gen, val_loader, latent_dim, device, num_samples=1000)
    
    print(f"Trial {trial.number} finished with FID: {fid:.4f}")
    return fid



# %%

print("Starting WGAN-GP Pipeline...")

# Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)
print(f"Best Trial: {study.best_params}")


# %%
# Final Training using best hyperparameters
#best = study.best_params
#print(f"Starting Final Training with params: {best}")

train_loader_full, test_loader = build_loaders(batch_size=128, train_limit=None)

gen = WGANGPGenerator(128, 64).to(device)
cri = WGANGPCritic(64).to(device)
gen.apply(weights_init)
cri.apply(weights_init)

model_params = {
    'latent_dim': 128,
    'ngf': 64,
    'lr': 0.0001
}

history = train_wgan_gp(
    gen, cri, train_loader_full, 128, 
    epochs=100, lr=model_params['lr'], 
    checkpoint_dir=Path('runs/wgan_gp/final_run'),
    val_loader=test_loader,
    print_progress=True
)

# Plot training loss
plt.figure(figsize=(10, 4))
plt.plot(history['g_loss'], label='Generator Loss')
plt.plot(history['c_loss'], label='Critic Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ArtBench-10 WGAN-GP - Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Save final comprehensive checkpoint
final_ckpt_path = Path('runs/wgan_gp/artbench_wgan_gp.pt')
final_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    'gen_state': gen.state_dict(),
    'cri_state': cri.state_dict(),
    'params': model_params,
    'history': history
}, final_ckpt_path)
print(f"✅ Final checkpoint saved to {final_ckpt_path}")

# %%
# ==========================================
# 1. CARREGANDO O MODELO TREINADO & INFERÊNCIA
# ==========================================
# %%
# ==========================================
# 1. CARREGANDO O MODELO TREINADO & INFERÊNCIA
# ==========================================
best_ckpt_path = Path('runs/wgan_gp/final_run/best_fid_model.pt')
if best_ckpt_path.exists():
    print(f"Loading checkpoint from {best_ckpt_path}...")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    
    # Rebuild model with same parameters used in training
    params = checkpoint.get('params', {'latent_dim': 128, 'ngf': 64})
    latent_dim = params['latent_dim']
    ngf = params['ngf']
    
    eval_gen = WGANGPGenerator(latent_dim, ngf).to(device)
    eval_gen.load_state_dict(checkpoint['gen_state'])
    eval_gen.eval()
    print("✅ Best model loaded for inference!")

    # 2. GERANDO AMOSTRAS
    print("Gerando amostras...")
    with torch.no_grad():
        z = torch.randn(25, latent_dim, device=device)
        samples = eval_gen(z)
    show_image_grid(samples, title='ArtBench-10 WGAN-GP - Amostras Geradas', n_show=25)

    # 3. INTERPOLAÇÃO NO ESPAÇO LATENTE
    print("\nInterpolação no espaço latente...")
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)
    n_interp = 10
    interp_images = []

    for alpha_val in torch.linspace(0, 1, n_interp):
        z_interp = (1 - alpha_val) * z1 + alpha_val * z2
        with torch.no_grad():
            img = eval_gen(z_interp)
        interp_images.append(img)

    interp_grid = torch.cat(interp_images, dim=0)
    show_image_grid(interp_grid, title='ArtBench-10 WGAN-GP - Interpolação Latente', n_show=n_interp)

    # 4. ROBUST EVALUATION
    run_robust_evaluation(eval_gen, test_loader, latent_dim, device, num_runs=10)
else:
    print(f"❌ Checkpoint {best_ckpt_path} not found.")

