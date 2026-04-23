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

# %%
# %pip install -q tqdm torchmetrics optuna
# %matplotlib inline

import sys
import random
from pathlib import Path
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import optuna

# Relative paths (run this script from student_start_pack/)
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


import requests, traceback

def notify(msg, title="Notebook"):
    requests.post("https://ntfy.sh/notebookIAGricardo",
        data=msg, headers={"Title": title, "Priority": "high"})



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
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

set_seed(42)
device = get_device()
print('Device:', device)

# %%
def build_loaders(
    dataset_name='artbench',
    batch_size=128,
    train_limit=10000,
    test_limit=None,
    data_root='IAGdata/artbench-10-python',
    num_workers=0,
):
    image_size = 32
    channels = 3
    
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    hf_ds = load_kaggle_artbench10_splits(KAGGLE_ROOT)
    train_ds = HFDatasetTorch(hf_ds["train"], transform=transform)
    test_ds = HFDatasetTorch(hf_ds["test"], transform=transform)
    class_names = hf_ds["train"].features["label"].names

    if train_limit is not None:
        train_ds = Subset(train_ds, list(range(min(train_limit, len(train_ds)))))
    if test_limit is not None:
        test_ds = Subset(test_ds, list(range(min(test_limit, len(test_ds)))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, channels, image_size, class_names

def denorm(x):
    return (x + 1.0) / 2.0

def show_image_grid(images, channels=3, title='Images', n_show=25, save_path=None):
    images = images[:n_show].detach().cpu()
    images = denorm(images).clamp(0, 1)

    n = images.size(0)
    grid = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(grid, grid, figsize=(grid * 1.6, grid * 1.6))
    axes = np.atleast_2d(axes)

    idx = 0
    for i in range(grid):
        for j in range(grid):
            ax = axes[i, j]
            ax.axis('off')
            if idx < n:
                ax.imshow(images[idx].permute(1, 2, 0))
            idx += 1
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


@torch.no_grad()
def evaluate_metrics(model, schedule, dataloader, device, num_samples=5000, use_ddim=True, ddim_steps=100):
    is_training = model.training
    model.eval()
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
    batch_size = 50
    while count < num_samples:
        current_bs = min(batch_size, num_samples - count)
        if use_ddim:
            fake_imgs = schedule.ddim_sample_loop(model, (current_bs, 3, 32, 32), ddim_steps=ddim_steps)
        else:
            fake_imgs = schedule.p_sample_loop(model, (current_bs, 3, 32, 32))
        fake_01 = denorm(fake_imgs).clamp(0, 1)
        fid.update(fake_01, real=False)
        kid.update(fake_01, real=False)
        count += current_bs

    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    fid.reset()
    kid.reset()
    if is_training:
        model.train()
    return fid_score, kid_mean.item(), kid_std.item()

# %% [markdown]
# ### Diffusion Components

# %%
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_schedule='cosine', beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        if beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(num_timesteps).to(device)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]).to(device), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_prod = self._get_index(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_prod = self._get_index(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise

    def _get_index(self, tensor, t, x_shape):
        out = tensor.gather(-1, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        model.eval()
        x = torch.randn(shape, device=self.device)
        for step in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            pred_noise = model(x, t)

            alpha_t = self.alphas[step]
            alpha_bar_t = self.alphas_cumprod[step]
            beta_t = self.betas[step]

            if step > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1.0 / torch.sqrt(alpha_t)) * (x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise) + torch.sqrt(beta_t) * noise
        return x

    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, ddim_steps=100, eta=0.0, x_init=None):
        """DDIM sampling for faster inference (Song et al., 2020)"""
        model.eval()
        # Create sub-sequence of timesteps
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))
        
        x = x_init.clone() if x_init is not None else torch.randn(shape, device=self.device)
        
        for i in range(len(timesteps)):
            t_cur = timesteps[i]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            
            t_batch = torch.full((shape[0],), t_cur, device=self.device, dtype=torch.long)
            pred_noise = model(x, t_batch)
            
            alpha_bar_t = self.alphas_cumprod[t_cur]
            alpha_bar_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=self.device)
            
            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            pred_x0 = pred_x0.clamp(-1, 1)
            
            # Compute variance
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * pred_noise
            
            noise = torch.randn_like(x) if t_cur > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise
        
        return x

# %%
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SelfAttention(nn.Module):
    def __init__(self, dim, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(min(num_groups, dim), dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.transpose(-1, -2) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-1, -2)).reshape(B, C, H, W)
        return x + self.proj(out)

class ResnetBlock(nn.Module):
    def __init__(self, dim, time_emb_dim, out_dim=None, num_groups=32):
        super().__init__()
        self.out_dim = out_dim or dim
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, self.out_dim))
        self.conv1 = nn.Conv2d(dim, self.out_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(self.out_dim, self.out_dim, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(num_groups, dim), dim)
        self.norm2 = nn.GroupNorm(min(num_groups, self.out_dim), self.out_dim)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(dim, self.out_dim, 1) if dim != self.out_dim else nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.mlp(time_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return self.shortcut(x) + h

class PixelUNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=128, num_groups=32):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        time_dim = model_channels * 4
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # 32 -> 16
        self.down1_res1 = ResnetBlock(model_channels, time_dim, num_groups=num_groups)
        self.down1_res2 = ResnetBlock(model_channels, time_dim, num_groups=num_groups)
        self.down1_pool = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        
        # 16 -> 8 (with attention at 16x16)
        self.down2_res1 = ResnetBlock(model_channels, time_dim, out_dim=model_channels * 2, num_groups=num_groups)
        self.down2_attn = SelfAttention(model_channels * 2, num_groups=num_groups)
        self.down2_res2 = ResnetBlock(model_channels * 2, time_dim, num_groups=num_groups)
        self.down2_pool = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        
        # Bottleneck at 8x8 (with attention)
        self.mid_res1 = ResnetBlock(model_channels * 2, time_dim, num_groups=num_groups)
        self.mid_attn = SelfAttention(model_channels * 2, num_groups=num_groups)
        self.mid_res2 = ResnetBlock(model_channels * 2, time_dim, num_groups=num_groups)
        
        # 8 -> 16 (with attention)
        self.up2_upsample = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1)
        self.up2_res1 = ResnetBlock(model_channels * 3, time_dim, out_dim=model_channels, num_groups=num_groups)
        self.up2_attn = SelfAttention(model_channels, num_groups=num_groups)
        self.up2_res2 = ResnetBlock(model_channels, time_dim, num_groups=num_groups)
        
        # 16 -> 32
        self.up1_upsample = nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1)
        self.up1_res1 = ResnetBlock(model_channels * 2, time_dim, out_dim=model_channels, num_groups=num_groups)
        self.up1_res2 = ResnetBlock(model_channels, time_dim, num_groups=num_groups)
        
        self.out_norm = nn.GroupNorm(min(num_groups, model_channels), model_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(model_channels, in_channels, 3, padding=1)
        
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        
        h_init = self.init_conv(x)
        
        # Down 1: 32 -> 16
        h1 = self.down1_res1(h_init, t_emb)
        h1 = self.down1_res2(h1, t_emb)
        h1_pool = self.down1_pool(h1)
        
        # Down 2: 16 -> 8 (with attention)
        h2 = self.down2_res1(h1_pool, t_emb)
        h2 = self.down2_attn(h2)
        h2 = self.down2_res2(h2, t_emb)
        h2_pool = self.down2_pool(h2)
        
        # Bottleneck at 8x8
        h_mid = self.mid_res1(h2_pool, t_emb)
        h_mid = self.mid_attn(h_mid)
        h_mid = self.mid_res2(h_mid, t_emb)
        
        # Up 2: 8 -> 16
        h_up2 = torch.cat([self.up2_upsample(h_mid), h2], dim=1)
        h_up2 = self.up2_res1(h_up2, t_emb)
        h_up2 = self.up2_attn(h_up2)
        h_up2 = self.up2_res2(h_up2, t_emb)
        
        # Up 1: 16 -> 32
        h_up1 = torch.cat([self.up1_upsample(h_up2), h1], dim=1)
        h_up1 = self.up1_res1(h_up1, t_emb)
        h_up1 = self.up1_res2(h_up1, t_emb)
        
        return self.out_conv(self.out_act(self.out_norm(h_up1)))

# %%
class EMA:
    """Exponential Moving Average of model weights (decay ~0.9999)"""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def get_model(self):
        return self.shadow


def save_checkpoint(model, ema, history, checkpoint_path, params):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'ema_state': ema.get_model().state_dict(),
        'params': params,
        'history': history
    }, checkpoint_path)
    print(f"✅ Checkpoint saved to {checkpoint_path}")


def train_diffusion(
    model, 
    loader, 
    schedule, 
    epochs=20, 
    lr=2e-4, 
    ema_decay=0.999, 
    grad_clip=1.0, 
    print_progress=True,
    val_loader=None,
    checkpoint_dir=None,
    save_interval=10,
    num_fid_samples=1500,
    model_params=None,
    trial=None # Added for Optuna pruning
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ema = EMA(model, decay=ema_decay)
    history = {'mse_loss': [], 'fid': []}
    model.train()

    best_fid = float('inf')

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / 'samples').mkdir(exist_ok=True)

    for epoch in range(epochs):
        running = 0.0
        n_batches = 0
        for x, _ in tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False, disable=not print_progress):
            x = x.to(device)
            t = torch.randint(0, schedule.num_timesteps, (x.size(0),), device=device).long()
            noise = torch.randn_like(x)
            x_t = schedule.q_sample(x_0=x, t=t, noise=noise)

            opt.zero_grad()
            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ema.update(model)

            running += loss.item()
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        history['mse_loss'].append(avg_loss)

        status_str = f'Epoch {epoch + 1:02d}/{epochs} | loss: {avg_loss:.6f}'

        # Optuna Pruning: Report progress and check if we should stop
        if trial is not None:
            trial.report(avg_loss, epoch)
            if trial.should_prune():
                if print_progress:
                    print(f" Trial pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()

        # Periodic Monitoring
# Visual Samples, FID, and Checkpoints
        if (epoch + 1) % save_interval == 0 or epoch == 0 or epoch == epochs - 1:
            # 1. Visual Samples (using EMA model and DDIM)
            if checkpoint_dir:
                with torch.no_grad():
                    ema_model = ema.get_model()
                    ema_model.eval()
                    samples = schedule.ddim_sample_loop(ema_model, (25, 3, 32, 32), ddim_steps=50)
                    sample_path = checkpoint_dir / 'samples' / f'epoch_{epoch+1:03d}.png'
                    show_image_grid(samples, channels=3, title=f'Epoch {epoch+1} (EMA/DDIM)', save_path=sample_path)
                    model.train()

            # 2. FID Metrics
            if val_loader:
                ema_model = ema.get_model()
                fid_score, kid_mean, _ = evaluate_metrics(ema_model, schedule, val_loader, device, num_samples=num_fid_samples, ddim_steps=50)
                history['fid'].append({'epoch': epoch + 1, 'fid': fid_score, 'kid': kid_mean})
                status_str += f" | FID: {fid_score:.2f}"
                
                # Save best model based on FID
                if fid_score < best_fid and checkpoint_dir:
                    best_fid = fid_score
                    save_checkpoint(model, ema, history, checkpoint_dir / 'best_fid_model.pt', model_params)
                    if print_progress:
                        print(f"   -> New best FID: {best_fid:.4f}. Saved best_fid_model.pt")

            # 3. Regular Checkpoint
            if checkpoint_dir:
                save_checkpoint(model, ema, history, checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pt', model_params)

        if print_progress:
            print(status_str)

    return history, best_fid, ema


# %%
# Load Data
artbench_train_loader, artbench_test_loader, artbench_channels, artbench_image_size, artbench_classes = build_loaders()

# %%
def objective(trial):
    # Optimizing only the Learning Rate (LR) as per strategy
    lr = trial.suggest_float("lr", 1e-4, 3e-4, log=True)
    
    # Fixed parameters recommended by the professor
    fixed_channels = 128
    fixed_schedule_type = 'cosine'
    
    print(f"\n>>> Trial {trial.number} | LR: {lr:.6f} | Channels: {fixed_channels} | Schedule: {fixed_schedule_type}")

    model = PixelUNet(in_channels=3, model_channels=fixed_channels).to(device)
    schedule = GaussianDiffusion(num_timesteps=1000, beta_schedule=fixed_schedule_type, device=device)
    
    # We use MSE loss for Optuna optimization
    try:
        history, _, _ = train_diffusion(
            model, 
            artbench_train_loader, 
            schedule, 
            epochs=30, 
            lr=lr, 
            print_progress=False,
            save_interval=31,
            trial=trial # Pass the trial for pruning
        )
        return history['mse_loss'][-1]
    except optuna.TrialPruned:
        raise # Re-raise to let Optuna handle it correctly

print("Starting Optuna Study (Optimizing Learning Rate with Pruning)...")
study = optuna.create_study(
    direction="minimize", 
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5) # Start pruning after 5 epochs
)
study.optimize(objective, n_trials=10, gc_after_trial=True)

print(f"\nMelhor LR encontrado: {study.best_params['lr']:.6f}")
print(f"Melhor Loss: {study.best_value:.6f}")

# %%
# Train best model with best params from Optuna
# best_params = study.best_params

best_lr = 0.000193
best_channels = 128
best_schedule_type = 'cosine'

best_pixel_model = PixelUNet(in_channels=3, model_channels=best_channels).to(device)
pixel_diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule=best_schedule_type, device=device)

# --- RE-LOAD FULL DATASET FOR FINAL TRAINING ---
print("Re-loading full ArtBench dataset (50k images)...")
artbench_train_loader_full, _, _, _, _ = build_loaders(
    dataset_name='artbench',
    batch_size=128,
    train_limit=None,
)

model_params = {
    'model_channels': best_channels, 
    'lr': best_lr, 
    'beta_schedule': best_schedule_type
}

history, _, pixel_ema = train_diffusion(
    best_pixel_model, 
    artbench_train_loader_full, 
    pixel_diffusion, 
    epochs=250, 
    lr=best_lr, 
    print_progress=True,
    val_loader=artbench_test_loader,
    checkpoint_dir='runs/diffusion/final_run',
    save_interval=10,
    model_params=model_params
)

# Use EMA model for sampling (better quality)
best_pixel_ema_model = pixel_ema.get_model()

# Plot training loss
plt.figure(figsize=(10, 4))
plt.plot(history['mse_loss'], label='MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ArtBench-10 Pixel Diffusion - Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Save final checkpoint
save_checkpoint(
    best_pixel_model, pixel_ema, history, 
    'runs/diffusion/artbench_pixel_diffusion.pt', 
    model_params
)



try:
    pass
    notify("✅ Finished successfully!")
except Exception as e:
    notify(f"❌ Failed: {traceback.format_exc()}", title="Notebook Error")

# %%
# ==========================================
# 1. CARREGANDO O MODELO TREINADO
# ==========================================
best_ckpt_path = Path('runs/diffusion/artbench_pixel_diffusion.pt')

if best_ckpt_path.exists():
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    loaded_params = ckpt['params']
    loaded_model = PixelUNet(in_channels=3, model_channels=loaded_params['model_channels']).to(device)
    
    
    # Prefer EMA weights if available (better sample quality)
    if 'ema_state' in ckpt:
        loaded_model.load_state_dict(ckpt['ema_state'])
        print("✅ EMA model loaded!")
    else:
        loaded_model.load_state_dict(ckpt['model_state'])
        print("✅ Model loaded (no EMA).")
    loaded_model.eval()
    
    beta_sched = loaded_params.get('beta_schedule', 'cosine')
    loaded_schedule = GaussianDiffusion(num_timesteps=1000, beta_schedule=beta_sched, device=device)
    print(f"Params: {loaded_params}")
else:
    print(f"❌ Checkpoint {best_ckpt_path} not found. Skipping inference.")
    loaded_model = None

# ==========================================
# 2. GERANDO AMOSTRAS (DDIM - faster)
# ==========================================
if loaded_model is not None:
    print("Gerando amostras via DDIM (100 steps)...")
    with torch.no_grad():
        samples = loaded_schedule.ddim_sample_loop(loaded_model, (25, 3, 32, 32), ddim_steps=100)
    show_image_grid(samples, channels=3, title='ArtBench-10 Pixel Diffusion - Amostras Geradas (DDIM)', n_show=25)

    # ==========================================
    # 3. INTERPOLAÇÃO NO ESPAÇO LATENTE (DDIM deterministic)
    # ==========================================
    print("\nInterpolação entre dois pontos de ruído (DDIM, eta=0)...")
    z1 = torch.randn(1, 3, 32, 32, device=device)
    z2 = torch.randn(1, 3, 32, 32, device=device)
    n_interp = 10
    interp_images = []

    for alpha_val in torch.linspace(0, 1, n_interp):
        z_interp = (1 - alpha_val) * z1 + alpha_val * z2
        with torch.no_grad():
            img = loaded_schedule.ddim_sample_loop(
                loaded_model, z_interp.shape, ddim_steps=100, eta=0.0, x_init=z_interp
            )
        interp_images.append(img)

    interp_grid = torch.cat(interp_images, dim=0)
    show_image_grid(interp_grid, channels=3, title='ArtBench-10 Pixel Diffusion - Interpolação Latente', n_show=n_interp)


# %%
def run_robust_evaluation(model, schedule, dataloader, device, num_runs=10, use_ddim=True, ddim_steps=100):
    fids, kids = [], []
    print(f"Starting robust evaluation ({num_runs} runs)...")
    for i in range(num_runs):
        print(f"\n--- Starting Run {i+1}/{num_runs} ---")
        set_seed(100 + i)
        f, k, _ = evaluate_metrics(model, schedule, dataloader, device, use_ddim=use_ddim, ddim_steps=ddim_steps)
        fids.append(f)
        kids.append(k)
        print(f"Run {i+1}/{num_runs} Completed | FID: {f:.4f} | KID: {k:.4f}")
    
    print("\n" + "="*30)
    print(f"FINAL RESULTS ({num_runs} Runs):")
    print(f"FID: {np.mean(fids):.4f} ± {np.std(fids):.4f}")
    print(f"KID: {np.mean(kids):.4f} ± {np.std(kids):.4f}")
    print("="*30)

# %%
ckpt_path = Path('runs/diffusion/final_run/checkpoint_epoch_100.pt')
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    params = ckpt['params']

    eval_model = PixelUNet(in_channels=3, model_channels=params['model_channels']).to(device)
    eval_model.load_state_dict(ckpt['ema_state'])
    eval_model.eval()

    pixel_diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule=params['beta_schedule'], device=device)

    run_robust_evaluation(eval_model, pixel_diffusion, artbench_test_loader, device, use_ddim=True, ddim_steps=100)


# %% [markdown]
# ## Latent Diffusion Model (LDM)

# %% [markdown]
# ### ConvVAE + treino VAE

# %%
class ConvVAE(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.latent_channels = latent_channels
 
        # Encoder: 3×32×32 → 128×8×8
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),   # → 64×16×16
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # → 128×8×8
            nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(),
        )
        # Cabeças mu e logvar espaciais: 128×8×8 → 4×8×8
        self.conv_mu     = nn.Conv2d(128, latent_channels, 1)
        self.conv_logvar = nn.Conv2d(128, latent_channels, 1)
 
        # Decoder: 4×8×8 → 3×32×32
        self.dec_input = nn.Conv2d(latent_channels, 128, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # → 64×16×16
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # → 32×32×32
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),  # saída em [0, 1]
        )
 
    def encode(self, x):
        h = self.encoder(x)
        return self.conv_mu(h), self.conv_logvar(h)
 
    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        return mu + eps * torch.exp(logvar * 0.5)
 
    def decode(self, z):
        return self.decoder(self.dec_input(z))
 
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# %%
def train_vae(vae, loader, epochs=50, lr=1e-3, beta_kl=1e-4,
              checkpoint_path='runs/ldm/vae.pt'):
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-2)
    history = []
    vae.train()
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
 
    for epoch in range(epochs):
        running = 0.0
        for x, _ in tqdm(loader, desc=f'VAE Epoch {epoch+1}/{epochs}', leave=False):
            x     = x.to(device)
            x_01  = denorm(x).clamp(0, 1)
 
            recon, mu, logvar = vae(x_01)
 
            recon_loss = F.mse_loss(recon, x_01)
            kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss       = recon_loss + beta_kl * kl_loss
 
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
 
        avg = running / len(loader)
        history.append(avg)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"VAE Epoch {epoch+1:02d}/{epochs} | Loss: {avg:.5f} | Recon: {recon_loss.item():.5f} | KL: {kl_loss.item():.5f}")
 
    torch.save(vae.state_dict(), checkpoint_path)
    print(f"VAE guardado em {checkpoint_path}")
    return history


# %%
vae = ConvVAE(latent_channels=4).to(device)
vae_history = train_vae(vae, artbench_train_loader, epochs=50, lr=2e-4)

# %%
vae.eval()
with torch.no_grad():
    sample_batch, _ = next(iter(artbench_test_loader))
    x_01 = denorm(sample_batch[:25]).clamp(0, 1).to(device)
    recons, _, _ = vae(x_01)
 
show_image_grid(recons * 2 - 1, title='VAE Reconstruções')
show_image_grid(sample_batch[:25], title='Originais')


# %% [markdown]
# ### UNet para o espaço latente (LDM UNet)

# %%
class LatentUNet(nn.Module):
    def __init__(self, in_channels=4, model_channels=128):
        super().__init__()

        self.channels = in_channels

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        time_dim = model_channels * 4

        # Entrada
        self.init_conv = nn.Conv2d(self.channels, model_channels, 3, padding=1)

        # Encoder: 8×8 → 4×4
        self.down_res1  = ResnetBlock(model_channels, time_dim)
        self.down_res2  = ResnetBlock(model_channels, time_dim)
        self.down_attn  = SelfAttention(model_channels)
        self.down_res3  = ResnetBlock(model_channels, time_dim)
        self.downsample = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)

        # Bottleneck: 4×4
        self.mid_res1 = ResnetBlock(model_channels, time_dim)
        self.mid_attn = SelfAttention(model_channels)
        self.mid_res2 = ResnetBlock(model_channels, time_dim)

        # Decoder: 4×4 → 8×8
        self.upsample  = nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1)
        self.up_res1   = ResnetBlock(model_channels * 2, time_dim, out_dim=model_channels)
        self.up_res2   = ResnetBlock(model_channels, time_dim)
        self.up_attn   = SelfAttention(model_channels)
        self.up_res3   = ResnetBlock(model_channels, time_dim)

        # Saída
        self.out_norm = nn.GroupNorm(min(32, model_channels), model_channels)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(model_channels, self.channels, 3, padding=1)

    def forward(self, z, t):
        t_emb = self.time_embed(t)

        x = self.init_conv(z)

        # Encoder
        x = self.down_res1(x, t_emb)
        x = self.down_res2(x, t_emb)
        x = self.down_attn(x)
        h = self.down_res3(x, t_emb)   
        x = self.downsample(h)          

        # Bottleneck
        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)

        # Decoder com skip connection
        x = self.upsample(x)            
        x = torch.cat([x, h], dim=1)   
        x = self.up_res1(x, t_emb)
        x = self.up_res2(x, t_emb)
        x = self.up_attn(x)
        x = self.up_res3(x, t_emb)

        return self.out_conv(self.out_act(self.out_norm(x)))  


# %% [markdown]
# ### Optuna — Search de LR para o LDM

# %%
# Latent scaling factor (standard LDM trick)
LATENT_SCALE = 0.18215

def train_ldm_full(
    model, vae, loader, schedule,
    epochs=20, lr=2e-4, ema_decay=0.999, grad_clip=1.0,
    print_progress=True, val_loader=None,
    checkpoint_dir=None, save_interval=10,
    num_fid_samples=1500, model_params=None, trial=None
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    ema = EMA(model, decay=ema_decay)
    history = {'mse_loss': [], 'fid': []}
    vae.eval()
    model.train()
    best_fid = float('inf')
 
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / 'samples').mkdir(exist_ok=True)
 
    for epoch in range(epochs):
        running = 0.0
        for x, _ in tqdm(loader, desc=f'LDM Epoch {epoch+1}/{epochs}',
                          leave=False, disable=not print_progress):
            x = x.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(denorm(x))
                z = vae.reparameterize(mu, logvar)  # (B, 4, 8, 8)
                z = z * LATENT_SCALE # Scale latents to unit variance
 
            t     = torch.randint(0, schedule.num_timesteps, (x.size(0),), device=device).long()
            noise = torch.randn_like(z)
            z_t   = schedule.q_sample(z, t, noise)
 
            opt.zero_grad()
            pred = model(z_t, t)
            loss = F.mse_loss(pred, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ema.update(model)
            running += loss.item()
 
        avg_loss = running / len(loader)
        history['mse_loss'].append(avg_loss)
        status_str = f'LDM Epoch {epoch+1:02d}/{epochs} | loss: {avg_loss:.6f}'
 
        # Optuna pruning
        if trial is not None:
            trial.report(avg_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
 
        if (epoch+1) % save_interval == 0 or epoch == 0 or epoch == epochs-1:
 
            # Amostras visuais
            if checkpoint_dir:
                ema_model = ema.get_model()
                ema_model.eval()
                with torch.no_grad():
                    z_rand = torch.randn(25, model.channels, 8, 8, device=device)
                    z_gen  = schedule.ddim_sample_loop(ema_model, z_rand.shape,
                                                        ddim_steps=50, x_init=z_rand)
                    imgs = vae.decode(z_gen / LATENT_SCALE) * 2 - 1  
                show_image_grid(imgs, title=f'LDM Epoch {epoch+1}',
                                save_path=checkpoint_dir / 'samples' / f'epoch_{epoch+1:03d}.png')
                model.train()
 
            # FID / KID
            if val_loader:
                ema_model = ema.get_model()
                fid_score, kid_mean, _ = evaluate_ldm_metrics(
                    ema_model, vae, schedule, val_loader, device,
                    num_samples=num_fid_samples
                )
                history['fid'].append({'epoch': epoch+1, 'fid': fid_score, 'kid': kid_mean})
                status_str += f' | FID: {fid_score:.2f}'
 
                if fid_score < best_fid and checkpoint_dir:
                    best_fid = fid_score
                    save_checkpoint(model, ema, history,
                                    checkpoint_dir / 'best_fid_model.pt', model_params)
                    if print_progress:
                        print(f'   -> Novo melhor FID: {best_fid:.4f}')
 
            #Checkpoint periódico
            if checkpoint_dir:
                save_checkpoint(model, ema, history,
                                checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pt',
                                model_params)
 
        if print_progress:
            print(status_str)
 
    return history, best_fid, ema


# %%
@torch.no_grad()
def evaluate_ldm_metrics(ldm_model, vae, schedule, dataloader, device,
                          num_samples=1500, ddim_steps=50):
    ldm_model.eval()
    vae.eval()
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=100, subsets=50, normalize=True).to(device)

    # Imagens reais
    count = 0
    for real_imgs, _ in dataloader:
        if count >= num_samples:
            break
        batch = real_imgs[:num_samples - count].to(device)
        fid.update(denorm(batch).clamp(0, 1), real=True)
        kid.update(denorm(batch).clamp(0, 1), real=True)
        count += batch.size(0)

    # Imagens geradas
    count = 0
    bs = 50
    while count < num_samples:
        cur_bs = min(bs, num_samples - count)
        z = torch.randn(cur_bs, ldm_model.channels, 8, 8, device=device)
        z_gen = schedule.ddim_sample_loop(ldm_model, z.shape,
                                           ddim_steps=ddim_steps, x_init=z)
        imgs = vae.decode(z_gen / LATENT_SCALE)
        # VAE termina com Sigmoid → já em [0,1], sem denorm
        fid.update(imgs.clamp(0, 1), real=False)
        kid.update(imgs.clamp(0, 1), real=False)
        count += cur_bs

    fid_val = fid.compute().item()
    kid_m, kid_s = kid.compute()
    fid.reset()
    kid.reset()
    return fid_val, kid_m.item(), kid_s.item()


# %%
def ldm_objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    print(f"\n>>> LDM Trial {trial.number} | LR: {lr:.6f}")

    model    = LatentUNet(in_channels=4, model_channels=128).to(device)
    schedule = GaussianDiffusion(num_timesteps=1000, beta_schedule='cosine', device=device)

    try:
        history, _, _ = train_ldm_full(
            model, vae, artbench_train_loader, schedule,
            epochs=30, lr=lr, print_progress=False,
            save_interval=31, trial=trial
        )
        return history['mse_loss'][-1]
    except optuna.TrialPruned:
        raise


# %%
ldm_study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)
ldm_study.optimize(ldm_objective, n_trials=10, gc_after_trial=True)

print(f"\nMelhor LR LDM: {ldm_study.best_params['lr']:.6f}")
print(f"Melhor Loss:   {ldm_study.best_value:.6f}")

# %% [markdown]
# ### Treino final LDM com melhor LR

# %%
best_ldm_lr = ldm_study.best_params['lr']

ldm_model    = LatentUNet(in_channels=4, model_channels=128).to(device)
ldm_schedule = GaussianDiffusion(num_timesteps=1000, beta_schedule='cosine', device=device)

# dataset completo 
artbench_train_full, _, _, _, _ = build_loaders(batch_size=128, train_limit=None)

ldm_params = {
    'model_channels': 128, 
    'lr': best_ldm_lr,
    'beta_schedule': 'cosine',
    'latent_channels': 4
}

ldm_history, _, ldm_ema = train_ldm_full(
    ldm_model, vae, artbench_train_full, ldm_schedule,
    epochs=100,  
    lr=best_ldm_lr,
    print_progress=True,
    val_loader=artbench_test_loader,
    checkpoint_dir='runs/ldm/final_run',
    save_interval=10,
    model_params=ldm_params
)

# Plot loss
plt.figure(figsize=(10, 4))
plt.plot(ldm_history['mse_loss'], label='LDM MSE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LDM — Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Guardar checkpoint final
save_checkpoint(ldm_model, ldm_ema, ldm_history,'runs/ldm/artbench_ldm.pt', ldm_params)


# %% [markdown]
# ### Inferência + Interpolação latente

# %%
ldm_ckpt_path = Path('runs/ldm/artbench_ldm.pt')
if ldm_ckpt_path.exists():
    ckpt = torch.load(ldm_ckpt_path, map_location=device, weights_only=True)
    p = ckpt['params']
    loaded_ldm = LatentUNet(in_channels=p.get('latent_channels', 4), model_channels=p['model_channels']).to(device)
    loaded_ldm.load_state_dict(ckpt['ema_state'])
    loaded_ldm.eval()
    loaded_ldm_schedule = GaussianDiffusion(
        num_timesteps=1000, beta_schedule=p['beta_schedule'], device=device
    )
    print(f"LDM carregado | params: {p}")
else:
    loaded_ldm = ldm_ema.get_model()
    loaded_ldm.eval()
    loaded_ldm_schedule = ldm_schedule

# Gerar amostras
vae.eval()
with torch.no_grad():
    z_rand = torch.randn(25, loaded_ldm.channels, 8, 8, device=device)
    z_gen  = loaded_ldm_schedule.ddim_sample_loop(
        loaded_ldm, z_rand.shape, ddim_steps=100, x_init=z_rand
    )
    imgs = vae.decode(z_gen / LATENT_SCALE) * 2 - 1  # [0,1] → [-1,1] para show_image_grid
show_image_grid(imgs, title='LDM — Amostras Geradas (DDIM 100 steps)', n_show=25)

# Interpolação no espaço latente
z1 = torch.randn(1, loaded_ldm.channels, 8, 8, device=device)
z2 = torch.randn(1, loaded_ldm.channels, 8, 8, device=device)
interp_imgs = []
with torch.no_grad():
    for alpha_val in torch.linspace(0, 1, 10):
        z_i   = (1 - alpha_val) * z1 + alpha_val * z2
        z_gen = loaded_ldm_schedule.ddim_sample_loop(
            loaded_ldm, z_i.shape, ddim_steps=100, eta=0.0, x_init=z_i
        )
        interp_imgs.append(vae.decode(z_gen / LATENT_SCALE) * 2 - 1)  # [0,1] → [-1,1]
show_image_grid(torch.cat(interp_imgs), title='LDM — Interpolação Latente', n_show=10)


# %% [markdown]
# ### Avaliação robusta

# %%
def run_robust_evaluation_ldm(ldm_model, vae, schedule, dataloader, device, num_runs=10, ddim_steps=100):
    fids, kids = [], []
    print(f"Robust evaluation LDM ({num_runs} runs)...")
    for i in range(num_runs):
        set_seed(100 + i)
        f, k, _ = evaluate_ldm_metrics(
            ldm_model, vae, schedule, dataloader, device, ddim_steps=ddim_steps
        )
        fids.append(f)
        kids.append(k)
        print(f"Run {i+1}/{num_runs} | FID: {f:.4f} | KID: {k:.4f}")

    print("\n" + "=" * 30)
    print(f"LDM FINAL ({num_runs} runs):")
    print(f"FID: {np.mean(fids):.4f} ± {np.std(fids):.4f}")
    print(f"KID: {np.mean(kids):.4f} ± {np.std(kids):.4f}")
    print("=" * 30)


run_robust_evaluation_ldm(loaded_ldm, vae, loaded_ldm_schedule, artbench_test_loader, device)
