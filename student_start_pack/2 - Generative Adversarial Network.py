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
# %pip install -q tqdm
# %matplotlib inline

import sys
import pickle
import os
import random
from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms as T

# Relative paths (run this script from student_start_pack/)
# Using absolute paths to avoid issues with different working directories
if "__file__" in globals():
    FILE_DIR = Path(__file__).resolve().parent
else:
    # Jupyter notebooks do not have __file__; assume we are in the notebook's directory
    FILE_DIR = Path.cwd()

PROJECT_ROOT = FILE_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
KAGGLE_ROOT = PROJECT_ROOT / 'ArtBench-10'

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from artbench_local_dataset import load_kaggle_artbench10_splits

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


# %% [markdown]
# reproducibility and device selection

# %%
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    return torch.device('cpu')


set_seed(42)
device = get_device()
print('Device:', device)


# %%
def build_loaders(
    dataset_name='mnist',
    batch_size=128,
    train_limit=30000,
    test_limit=5000,
    data_root='data',
    num_workers=0,
):
    dataset_name = dataset_name.lower()

    if dataset_name == 'mnist':
        channels = 1
        image_size = 32
        transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])
        train_ds = datasets.MNIST(data_root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(data_root, train=False, download=True, transform=transform)
        class_names = [str(i) for i in range(10)]
    elif dataset_name in ('cifar', 'cifar10'):
        channels = 3
        image_size = 32
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_ds = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(data_root, train=False, download=True, transform=transform)
        class_names = train_ds.classes
    elif dataset_name == 'artbench':
        channels = 3
        image_size = 32
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
    else:
        raise ValueError("dataset_name must be 'mnist', 'cifar10' or 'artbench'.")

    if train_limit is not None:
        train_ds = Subset(train_ds, list(range(min(train_limit, len(train_ds)))))
    if test_limit is not None:
        test_ds = Subset(test_ds, list(range(min(test_limit, len(test_ds)))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, channels, image_size, class_names


# %%
def denorm(x):
    return (x + 1.0) / 2.0


def show_image_grid(images, channels, title='Images', n_show=25):
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
                if channels == 1:
                    ax.imshow(images[idx, 0], cmap='gray', vmin=0, vmax=1)
                else:
                    ax.imshow(images[idx].permute(1, 2, 0))
            idx += 1

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# %%
class DCGenerator(nn.Module):
    def __init__(self, latent_dim=100, image_channels=3, ngf=64):
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
            nn.ConvTranspose2d(ngf, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)


class DCDiscriminator(nn.Module):
    def __init__(self, image_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(image_channels, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def init_dcgan_weights(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def plot_gan_losses(history, title='GAN losses'):
    plt.figure(figsize=(7, 4))
    plt.plot(history['d_loss'], label='Discriminator loss')
    plt.plot(history['g_loss'], label='Generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def save_checkpoint(generator, discriminator, history, checkpoint_path, latent_dim, channels, image_size):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'history': history,
            'config': {
                'latent_dim': latent_dim,
                'channels': channels,
                'image_size': image_size,
            },
        },
        checkpoint_path,
    )
    print('Saved checkpoint to', checkpoint_path)


@torch.no_grad()
def load_dcgan_generator_for_inference(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt['config']
    generator = DCGenerator(latent_dim=cfg['latent_dim'], image_channels=cfg['channels']).to(device)
    generator.load_state_dict(ckpt['generator'])
    generator.eval()
    return generator, cfg, ckpt.get('history', None)



# %%
def train_gan(
    generator,
    discriminator,
    loader,
    latent_dim,
    epochs=20,
    lr=2e-4,
    d_steps=2,
    beta1=0.5,
    beta2=0.99,
    trial=None,
    print_progress=True,
    patience=10,
    use_early_stopping=True  # NOVA VARIÁVEL AQUI
):
    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    history = {'g_loss': [], 'd_loss': []}
    generator.train()
    discriminator.train()
    
    best_g_loss = float('inf')
    patience_counter = 0
    best_generator_weights = None
    best_discriminator_weights = None 

    for epoch in range(epochs):
        g_running = 0.0
        d_running = 0.0
        n_batches = 0

        for real, _ in tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            real = real.to(device)
            bs = real.size(0)

            real_targets = torch.ones(bs, 1, device=device)
            fake_targets = torch.zeros(bs, 1, device=device)

            # TODO START - Discriminator update
            d_loss = None
            for _ in range(d_steps):
                opt_d.zero_grad(set_to_none=True)
                discriminator_real_outputs = discriminator(real)
                d_loss_real = criterion(discriminator_real_outputs, real_targets)

                noise = torch.randn(bs, latent_dim, device=device)
                fake = generator(noise)
                discriminator_fake_outputs = discriminator(fake.detach())
                d_loss_fake = criterion(discriminator_fake_outputs, fake_targets)

                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                d_loss.backward()
                opt_d.step()
            # TODO END

            if d_loss is None:
                raise NotImplementedError('Implement Discriminator update in train_gan.')

            # --- TODO START - Generator update ---
            opt_g.zero_grad(set_to_none=True)

            noise = torch.randn(bs, latent_dim, device=device)
            fake = generator(noise)
            discriminator_outputs_on_fake = discriminator(fake)

            # Non-saturating generator objective: maximize log D(G(z)).
            g_loss = criterion(discriminator_outputs_on_fake, real_targets)

            g_loss.backward()
            opt_g.step()
            # TODO END

            if g_loss is None:
                raise NotImplementedError('Implement Generator update in train_gan.')

            # TODO START - Bookkeeping
            g_running += g_loss.item()
            d_running += d_loss.item()
            n_batches += 1
            # TODO END
            
        # Calcula as médias da época atual
        epoch_g_loss = g_running / max(n_batches, 1)
        epoch_d_loss = d_running / max(n_batches, 1)

        history['g_loss'].append(epoch_g_loss)
        history['d_loss'].append(epoch_d_loss)

        if trial is not None:
            trial.report(epoch_g_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if print_progress:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"D loss: {history['d_loss'][-1]:.4f} | "
                f"G loss: {history['g_loss'][-1]:.4f}"
            )
            
        # --- LÓGICA DO EARLY STOPPING ---
        if use_early_stopping:
            warmup_epochs = 10
            
            if epoch < warmup_epochs:
                best_g_loss = epoch_g_loss
                best_generator_weights = copy.deepcopy(generator.state_dict())
                best_discriminator_weights = copy.deepcopy(discriminator.state_dict())
                if print_progress:
                    print(f"   -> Aquecimento ({epoch+1}/{warmup_epochs}). Definindo linha de base: {best_g_loss:.4f}")
            else:
                if epoch_g_loss < best_g_loss:
                    best_g_loss = epoch_g_loss
                    patience_counter = 0
                    best_generator_weights = copy.deepcopy(generator.state_dict())
                    best_discriminator_weights = copy.deepcopy(discriminator.state_dict())
                    if print_progress:
                        print(f"   -> Nova melhor G Loss: {best_g_loss:.4f}. Resetando paciência.")
                else:
                    patience_counter += 1
                    if print_progress:
                        print(f"   -> Sem melhora. Paciência: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    if print_progress:
                        print(f"\n[Early Stopping] Treinamento interrompido na época {epoch + 1}!")
                    break
        else:
            # Se não estamos a usar early stopping, a "melhor loss" é simplesmente a última para não dar erro no return
            best_g_loss = epoch_g_loss
        # --------------------------------

    # --- RESTAURAR OS MELHORES PESOS (Apenas se o Early Stopping estiver ativo) ---
    if use_early_stopping and best_generator_weights is not None:
        generator.load_state_dict(best_generator_weights)
        discriminator.load_state_dict(best_discriminator_weights)

    return history, best_g_loss


# %%
artbench_train_loader, artbench_test_loader, artbench_channels, artbench_image_size, artbench_classes = build_loaders(
    dataset_name='artbench',
    batch_size=128,
    train_limit=10000, # 20% of the training data for development
    test_limit=None, # Use all test data for evaluation
    data_root='IAGdata/artbench-10-python',
)

x_artbench, _ = next(iter(artbench_train_loader))
show_image_grid(x_artbench, channels=artbench_channels, title='ArtBench-10 real samples (32x32)', n_show=25)

# %%
# TODO START
artbench_latent_dim = 100
artbench_epochs = 50 # Increased epochs for better results on ArtBench
artbench_lr = 2e-4
artbench_ckpt = Path('runs/dcgan/artbench_dcgan.pt')

artbench_generator = DCGenerator(latent_dim=artbench_latent_dim, image_channels=artbench_channels).to(device)
artbench_discriminator = DCDiscriminator(image_channels=artbench_channels).to(device)
# TODO END

if artbench_generator is None or artbench_discriminator is None:
    raise NotImplementedError('Instantiate ArtBench DCGAN models inside TODO block.')

# TODO START
# Recommended: initialize DCGAN weights
artbench_generator.apply(init_dcgan_weights)
artbench_discriminator.apply(init_dcgan_weights)
# TODO END

artbench_history, _ = train_gan(
    generator=artbench_generator,
    discriminator=artbench_discriminator,
    loader=artbench_train_loader,
    latent_dim=artbench_latent_dim,
    epochs=artbench_epochs,
    lr=artbench_lr,
    print_progress=True
)

plot_gan_losses(artbench_history, title='ArtBench-10 DCGAN losses')

save_checkpoint(
    generator=artbench_generator,
    discriminator=artbench_discriminator,
    history=artbench_history,
    checkpoint_path=artbench_ckpt,
    latent_dim=artbench_latent_dim,
    channels=artbench_channels,
    image_size=artbench_image_size,
)


# %%
@torch.no_grad()
def run_inference(generator, latent_dim, channels, n_samples=25, seed=123, title='Generator inference'):
    # TODO START
    # 1) set torch seed para garantir que os resultados possam ser reproduzidos
    torch.manual_seed(seed)
    
    # 2) sample z: Criamos o ruído aleatório. O formato é [Quantidade_de_imagens, Tamanho_do_vetor_latente]
    z = torch.randn(n_samples, latent_dim, device=device)
    
    # 3) generate fake images: Passamos o ruído pelo Gerador para obter as imagens
    fake = generator(z)
    # TODO END

    if z is None or fake is None:
        raise NotImplementedError('Implement run_inference for DCGAN.')

    show_image_grid(fake, channels=channels, title=title, n_show=n_samples)


@torch.no_grad()
def latent_walk(generator, latent_dim, channels, steps=10, title='Latent interpolation'):
    # TODO START
    # 1) sample z0 and z1: Sorteamos os nossos pontos de "partida" e "chegada" no espaço latente
    z0 = torch.randn(1, latent_dim, device=device)
    z1 = torch.randn(1, latent_dim, device=device)
    
    # 2) interpolate with alpha in [0, 1]
    # linspace cria um tensor com 'steps' valores, espaçados igualmente entre 0 e 1
    alphas = torch.linspace(0, 1, steps, device=device)
    
    # Fazemos a interpolação linear para cada alpha e usamos torch.cat para empilhar tudo
    # em um único tensor 'z' com a forma [steps, latent_dim]
    z = torch.cat([(1 - a) * z0 + a * z1 for a in alphas], dim=0)
    
    # 3) generate fake images: Transformamos essa "caminhada" em imagens visuais
    fake = generator(z)
    # TODO END

    if z is None or fake is None:
        raise NotImplementedError('Implement latent_walk for DCGAN.')

    show_image_grid(fake, channels=channels, title=title, n_show=steps)


# %%
# ==========================================
# 1. CARREGANDO O MODELO TREINADO
# ==========================================
# TODO START
# usamos a função auxiliar para carregar o Gerador e suas configurações (cfg) do disco
if artbench_ckpt.exists():
    artbench_gen_infer, artbench_cfg, _ = load_dcgan_generator_for_inference(artbench_ckpt)
else:
    print(f"Checkpoint {artbench_ckpt} not found. Skipping inference.")
    artbench_gen_infer = None
    artbench_cfg = None
# TODO END

if artbench_gen_infer is not None and artbench_cfg is not None:
    # ==========================================
    # 2. EXECUTANDO A GERAÇÃO DE IMAGENS
    # ==========================================
    # TODO START
    print("Gerando amostras aleatórias do espaço latente...")
    # Extraímos os valores do dicionário de configuração (cfg) que foi salvo no checkpoint
    run_inference(
        generator=artbench_gen_infer,
        latent_dim=artbench_cfg['latent_dim'],
        channels=artbench_cfg['channels'],
        n_samples=25,
        title='ArtBench-10 DCGAN - Amostras Aleatórias'
    )

    print("\nCaminhando suavemente pelo espaço latente...")
    latent_walk(
        generator=artbench_gen_infer,
        latent_dim=artbench_cfg['latent_dim'],
        channels=artbench_cfg['channels'],
        steps=100, # Vai gerar 10 imagens mostrando a transição de uma obra para outra
        title='ArtBench-10 DCGAN - Interpolação Latente'
    )
    # TODO END

# %%
import optuna
import pandas as pd
from pathlib import Path

def objective(trial):
    # Busca mais alinhada à teoria DCGAN
    lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
    z_dim = trial.suggest_categorical("latent_dim", [64, 100, 128])
    d_steps = trial.suggest_categorical("d_steps", [1, 2, 3])
    beta1 = trial.suggest_categorical("beta1", [0.0, 0.5])
    feature_maps = trial.suggest_categorical("feature_maps", [64, 96])

    print(
        f"\n>>> Trial {trial.number} | LR: {lr:.6f} | Z: {z_dim} | "
        f"D steps: {d_steps} | beta1: {beta1} | FM: {feature_maps}"
    )

    generator = DCGenerator(
        latent_dim=z_dim,
        image_channels=artbench_channels,
        ngf=feature_maps,
    ).to(device)
    discriminator = DCDiscriminator(
        image_channels=artbench_channels,
        ndf=feature_maps,
    ).to(device)
    generator.apply(init_dcgan_weights)
    discriminator.apply(init_dcgan_weights)

    history, best_g_loss = train_gan(
        generator=generator,
        discriminator=discriminator,
        loader=artbench_train_loader,
        latent_dim=z_dim,
        epochs=30,
        lr=lr,
        d_steps=d_steps,
        beta1=beta1,
        beta2=0.99,
        trial=trial,
        print_progress=False,
    )

    return best_g_loss

# --- EXECUTAR O OPTUNA ---
print("Iniciando otimização com Optuna...")

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)

study.optimize(objective, n_trials=20, gc_after_trial=True)

print("\n==== RESULTADOS DO OPTUNA ====")
print(f"Melhor Trial: {study.best_trial.number}")
print(f"Melhor Loss do Gerador: {study.best_value:.4f}")
print(f"Melhores Hiperparâmetros: {study.best_params}")

# --- SALVAR RESULTADOS EM ARQUIVOS ---
# 1. Garante que a pasta existe
output_dir = Path("runs/dcgan")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Salva o histórico completo de todas as tentativas em CSV
df_results = study.trials_dataframe()
csv_path = output_dir / "optuna_trials_history.csv"
df_results.to_csv(csv_path, index=False)
print(f"\nHistórico completo (CSV) salvo em: {csv_path}")

# 3. Salva um resumo rápido (TXT) apenas com os melhores valores
summary_path = output_dir / "optuna_best_summary.txt"
with open(summary_path, "w") as f:
    f.write("==== RESULTADOS FINAIS OPTUNA - DCGAN ====\n")
    f.write(f"Melhor Trial: {study.best_trial.number}\n")
    f.write(f"Melhor Loss do Gerador: {study.best_value:.4f}\n")
    f.write(f"Melhores Hiperparâmetros: {study.best_params}\n")
print(f"Resumo dos melhores parâmetros (TXT) salvo em: {summary_path}")

# %% [markdown]
# Melhores Hiperpar�metros: {'lr': 0.0003496206556181923, 'latent_dim': 64, 'd_steps': 1, 'beta1': 0.0, 'feature_maps': 64}
#

# %%
# Train with the best hyperparameters found by Optuna
best_lr = 0.0003496206556181923
best_z_dim = 64
best_d_steps = 1
best_beta1 = 0.0
best_feature_maps = 64

# Instanciamos os modelos passando os feature_maps (ngf e ndf)
best_generator = DCGenerator(
    latent_dim=best_z_dim, 
    image_channels=artbench_channels,
    ngf=best_feature_maps
).to(device)

best_discriminator = DCDiscriminator(
    image_channels=artbench_channels,
    ndf=best_feature_maps
).to(device)

best_generator.apply(init_dcgan_weights)
best_discriminator.apply(init_dcgan_weights)

best_history, _ = train_gan(
    generator=best_generator,
    discriminator=best_discriminator,
    loader=artbench_train_loader,
    latent_dim=best_z_dim,
    epochs=50, 
    lr=best_lr,
    d_steps=best_d_steps, # Adicionado
    beta1=best_beta1,     # Adicionado
    beta2=0.99,           # Mantendo o valor padrão usado no Optuna
    use_early_stopping=False,
    print_progress=True
)

plot_gan_losses(best_history, title='ArtBench-10 DCGAN losses (Best Hyperparameters)')

# Salva o modelo treinado com os melhores hiperparâmetros
save_checkpoint(
    generator=best_generator,
    discriminator=best_discriminator,
    history=best_history,
    checkpoint_path='runs/dcgan/artbench_dcgan_best_optuna.pt',
    latent_dim=best_z_dim,
    channels=artbench_channels,
    image_size=artbench_image_size,
)

# %%
# ==========================================
# 1. CARREGANDO O MODELO TREINADO
# ==========================================
# Define o caminho exato do modelo campeão que acabamos de salvar
best_ckpt_path = Path('runs/dcgan/artbench_dcgan_best_optuna.pt')

# Usamos a função auxiliar para carregar o Gerador e suas configurações (cfg) do disco
if best_ckpt_path.exists():
    artbench_gen_infer, artbench_cfg, _ = load_dcgan_generator_for_inference(best_ckpt_path)
    print("✅ Modelo campeão carregado com sucesso!")
else:
    print(f" Checkpoint {best_ckpt_path} not found. Skipping inference.")
    artbench_gen_infer = None
    artbench_cfg = None

if artbench_gen_infer is not None and artbench_cfg is not None:
    # ==========================================
    # 2. EXECUTANDO A GERAÇÃO DE IMAGENS
    # ==========================================
    # TODO START
    print("Gerando amostras aleatórias do espaço latente...")
    # Extraímos os valores do dicionário de configuração (cfg) que foi salvo no checkpoint
    run_inference(
        generator=artbench_gen_infer,
        latent_dim=artbench_cfg['latent_dim'],
        channels=artbench_cfg['channels'],
        n_samples=25,
        title='ArtBench-10 DCGAN - Amostras Aleatórias'
    )

    print("\nCaminhando suavemente pelo espaço latente...")
    latent_walk(
        generator=artbench_gen_infer,
        latent_dim=artbench_cfg['latent_dim'],
        channels=artbench_cfg['channels'],
        steps=100, # Vai gerar 10 imagens mostrando a transição de uma obra para outra
        title='ArtBench-10 DCGAN - Interpolação Latente'
    )
    # TODO END

# %%
import requests, traceback

def notify(msg, title="Notebook"):
    requests.post("https://ntfy.sh/notebookIAGricardo",
        data=msg, headers={"Title": title, "Priority": "high"})

try:
    pass
    notify("✅ Finished successfully!")
except Exception as e:
    notify(f"❌ Failed: {traceback.format_exc()}", title="Notebook Error")

# %%
import torch
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm.auto import tqdm

@torch.no_grad()
def evaluate_metrics(generator, dataloader, latent_dim, device, num_samples=5000):
    """
    Extrai features de 5000 imagens reais e 5000 imagens geradas
    para calcular o FID e o KID de uma única execução.
    """
    generator.eval()
    
    # Inicializando as métricas. 
    # normalize=True avisa que os tensores estarão no intervalo [0, 1] float
    # KID configurado para 50 subsets de tamanho 100, como exige o enunciado
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=100, subsets=50, normalize=True).to(device)

    print("Extraindo features de 5000 imagens REAIS...")
    real_count = 0
    for real_imgs, _ in dataloader:
        if real_count >= num_samples:
            break
            
        batch_size = real_imgs.size(0)
        # Corta o batch se for ultrapassar 5000 imagens
        if real_count + batch_size > num_samples:
            real_imgs = real_imgs[:num_samples - real_count]
            batch_size = real_imgs.size(0)

        real_imgs = real_imgs.to(device)
        # O dataloader normaliza as imagens para [-1, 1]. O torchmetrics espera [0, 1]
        real_imgs_01 = ((real_imgs + 1.0) / 2.0).clamp(0, 1)

        fid.update(real_imgs_01, real=True)
        kid.update(real_imgs_01, real=True)
        
        real_count += batch_size

    print("Gerando e extraindo features de 5000 imagens FALSAS...")
    fake_count = 0
    batch_size = 100 # Gera de 100 em 100 para não sobrecarregar a memória da GPU
    while fake_count < num_samples:
        current_batch_size = min(batch_size, num_samples - fake_count)
        
        z = torch.randn(current_batch_size, latent_dim, device=device)
        fake_imgs = generator(z)
        
        # Mapeia [-1, 1] para [0, 1]
        fake_imgs_01 = ((fake_imgs + 1.0) / 2.0).clamp(0, 1)

        fid.update(fake_imgs_01, real=False)
        kid.update(fake_imgs_01, real=False)
        
        fake_count += current_batch_size

    # Computar os resultados
    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    
    # Limpa as métricas da memória para a próxima execução
    fid.reset()
    kid.reset()

    return fid_score, kid_mean.item(), kid_std.item()


def run_robust_evaluation(generator, dataloader, latent_dim, device, num_runs=10):
    """
    Roda a avaliação múltiplas vezes com seeds diferentes para robustez estatística
    """
    fid_scores = []
    kid_means = []
    
    for i in range(num_runs):
        # Altera a seed a cada repetição 
        seed = 100 + i
        torch.manual_seed(seed)
        
        print(f"\n--- Iniciando Avaliação {i+1}/{num_runs} (Seed: {seed}) ---")
        
        fid_score, kid_mean, _ = evaluate_metrics(
            generator=generator, 
            dataloader=dataloader, 
            latent_dim=latent_dim, 
            device=device,
            num_samples=5000
        )
        
        fid_scores.append(fid_score)
        kid_means.append(kid_mean)
        print(f"Run {i+1} finalizada | FID: {fid_score:.4f} | KID (média): {kid_mean:.4f}")

    # Calcula Média e Desvio Padrão das 10 repetições
    final_fid_mean = np.mean(fid_scores)
    final_fid_std = np.std(fid_scores)
    final_kid_mean = np.mean(kid_means)
    final_kid_std = np.std(kid_means)

    print("\n" + "="*50)
    print("RESULTADOS FINAIS PARA O RELATÓRIO (10 Repetições):")
    print(f"FID: {final_fid_mean:.4f} ± {final_fid_std:.4f}")
    print(f"KID: {final_kid_mean:.4f} ± {final_kid_std:.4f}")
    print("="*50)


# %%
if artbench_gen_infer is not None and artbench_cfg is not None:
    print("\nIniciando Protocolo de Avaliação Quantitativa (Robustez Estatística)...")
    run_robust_evaluation(
        generator=artbench_gen_infer,
        dataloader=artbench_test_loader,  # Usar o loader de teste ou treino dependendo da escolha
        latent_dim=artbench_cfg['latent_dim'],
        device=device,
        num_runs=10 
    )

# %%
import requests, traceback

def notify(msg, title="Notebook"):
    requests.post("https://ntfy.sh/notebookIAGricardo",
        data=msg, headers={"Title": title, "Priority": "high"})

try:
    pass
    notify("✅ Finished successfully!")
except Exception as e:
    notify(f"❌ Failed: {traceback.format_exc()}", title="Notebook Error")
