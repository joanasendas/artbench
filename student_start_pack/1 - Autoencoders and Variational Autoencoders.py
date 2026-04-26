import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from pathlib import Path
from tqdm.auto import tqdm

# Device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

#MODELOS
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=(1024, 512), latent_dim=64):
        super(DenseAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        #Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            nn.ReLU(), #minimize the errors between the layers
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], self.latent_dim)
            )
        
        
        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.input_dim),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.decoder(self.encoder(x))
    
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        # TODO START
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU()
        )
        self.enc_fc = nn.Linear(128*8*8, self.latent_dim)
        self.dec_fc = nn.Linear(self.latent_dim, 128*8*8)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        #raise NotImplementedError('Implement ConvAutoencoder layers from the recipe')
        # TODO END

    def encode(self, x):
        h = self.enc_conv(x)
        return self.enc_fc(h.view(h.size(0), -1))

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 128, 8, 8)
        return self.dec_conv(h)

    def forward(self, x):
        return self.decode(self.encode(x))

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # TODO START
        # Add mean and log-variance heads from flattened conv features.
        self.fc_mu = nn.Linear(128*8*8, latent_dim)
        self.fc_logvar = nn.Linear(128*8*8, latent_dim)
        #raise NotImplementedError('Implement ConvVAE latent heads')
        # TODO END

        self.dec_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # TODO START
        eps = torch.randn_like(logvar)
        z = mu + eps * torch.exp(logvar * 0.5)
        return z
        #raise NotImplementedError('Implement reparameterization')
        # TODO END

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 128, 8, 8)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

class ConvVAE2(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.flatten_dim = 512 * 4 * 4

        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

      
        self.dec_fc = nn.Linear(latent_dim, self.flatten_dim)
        
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),   
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        z = mu + eps * torch.exp(logvar * 0.5)
        return z

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 512, 4, 4)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar
    
class CondVAE(nn.Module):
    def __init__(self, latent_dim=64, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Adicionamos num_classes ao tamanho de entrada
        self.fc_mu = nn.Linear(128*8*8 + num_classes, latent_dim)
        self.fc_logvar = nn.Linear(128*8*8 + num_classes, latent_dim)

        #O decoder recebe o espaço latente + num_classes
        self.dec_fc = nn.Linear(latent_dim + num_classes, 128 * 8 * 8)
        
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x, c):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        # Concatenamos as features com a condição 'c'
        h = torch.cat([h, c], dim=1) 
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        z = mu + eps * torch.exp(logvar * 0.5)
        return z

    def decode(self, z, c):
        # Concatenamos o vetor latente com a condição 'c'
        z = torch.cat([z, c], dim=1)
        h = self.dec_fc(z).view(-1, 128, 8, 8)
        return self.dec_conv(h)

    def forward(self, x, c):
        # Passamos 'c' adiante em todas as etapas
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z, c)
        return xhat, mu, logvar
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_permuted.view(-1, self.embedding_dim)
        
        
        distances = (torch.sum(z_flattened**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(z_permuted.shape)
        
        #z_q = z_permuted + (z_q - z_permuted).detach()
        z_q_st = z_permuted + (z_q - z_permuted).detach()

        loss = F.mse_loss(z_q.detach(), z_permuted) + self.commitment_cost * F.mse_loss(z_q, z_permuted.detach())

        z_q_st = z_q_st.permute(0, 3, 1, 2).contiguous()
        
        return z_q_st, loss
    
    
class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x): 
        # 1. Encode
        z = self.encoder(x)

        # 2. Quantize
        z_q, quantization_loss = self.quantizer(z)

        # 3. Decode
        x_hat = self.decoder(z_q)
        
        return x_hat, quantization_loss

class HierarchicalVQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128):
        super().__init__()

        self.encoder_bottom = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
        )

        self.encoder_top = nn.Sequential(
            nn.Conv2d(128, 128, 4, stride=2, padding=1), # 8x8 -> 4x4
            nn.ReLU(),
        )

        self.quantizer_top = VectorQuantizer(num_embeddings, embedding_dim)
        self.quantizer_bottom = VectorQuantizer(num_embeddings, embedding_dim)

        self.decoder_top = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, 4, stride=2, padding=1), # 4x4 -> 8x8
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim + 128, 64, 4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1), # 16x16 -> 32x32
            nn.Sigmoid(),
        )

    def forward(self, x):
        z_bottom = self.encoder_bottom(x)     # (B,128,8,8)
        z_top = self.encoder_top(z_bottom)    # (B,128,4,4)

        z_top_q, loss_top = self.quantizer_top(z_top)

        z_top_up = self.decoder_top(z_top_q)  # (B,128,8,8)

        z_bottom_q, loss_bottom = self.quantizer_bottom(z_bottom)

        z_combined = torch.cat([z_bottom_q, z_top_up], dim=1)

        x_hat = self.decoder(z_combined)

        quant_loss = loss_top + loss_bottom

        return x_hat, quant_loss

#FUNÇÔES DE TREINO E AVALIAÇÃO 

def vae_loss(xhat, x, mu, logvar, beta=0.7):
    # TODO START
    recon = F.binary_cross_entropy(xhat, x, reduction='sum')/x.size(0) 
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/x.size(0)
    loss = recon + beta * kl
    return loss, recon, kl
    #raise NotImplementedError('Implement VAE loss')
    # TODO END

def vae_loss_mse(xhat, x, mu, logvar, beta=0.1099):
    recon = F.mse_loss(xhat, x, reduction='sum') / x.size(0)
    kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl, recon, kl

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def train_autoencoder(model, train_loader, val_loader, optimizer, save_path, epochs=20, flatten_input=True):
    model.train()
    history = []

    early_stopper = EarlyStopper(patience=10, min_delta=0.001)
    best_val_loss = float('inf')

    for ep in range(epochs):
        run = 0.0
        ## ALTERAÇÃO: o ArtBench retorna (imagem, label, índice)
        for x, _, _ in tqdm(train_loader, leave=False):
            x = x.to(device) 
            xin = x.view(x.size(0), -1) if flatten_input else x
            # TODO START
            xhat = model(xin)
            loss = F.binary_cross_entropy(xhat, xin, reduction='sum')/x.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # TODO END
            run += loss.item() * x.size(0)

        epoch_recon = run / len(train_loader.dataset)

        val_metrics = evaluate_autoencoder(model, val_loader, flatten_input)
        val_loss = val_metrics['bce'] # escolher entre 'bce' e 'rmse'

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path) # Guarda no caminho indicado

        history.append({'train_recon_bce': epoch_recon, 'val_recon_bce': val_loss})
        print(f'Epoch {ep+1}/{epochs} | train_recon_bce={epoch_recon:.4f} | val_recon_bce={val_loss:.4f}')

        if early_stopper.early_stop(val_loss):
            print(f'Early stopping ativado na época {ep+1}')
            break
    return history


def evaluate_autoencoder(model, loader, flatten_input=True):
    model.eval()
    tb, tm, ta, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            xin = x.view(x.size(0), -1) if flatten_input else x
            xhat = model(xin)
            b = x.size(0)
            # TODO START
            tb += F.binary_cross_entropy(xhat, xin, reduction='sum').item()
            tm += F.mse_loss(xhat, xin, reduction='sum').item()
            ta += F.l1_loss(xhat, xin, reduction='sum').item()  
            n += b
            # TODO END
    numel = xin[0].numel()
    return {'bce': tb/n, 'mse': tm/(n*numel), 'mae': ta/(n*numel), 'rmse': math.sqrt(tm / (n * numel)),}

def train_vae(model, train_loader, val_loader, save_path, optimizer, epochs=20, beta=0.7, flatten_input=False):
    model.train()
    hist = []
    best_val_loss = float('inf')
    early_stopper = EarlyStopper(patience=10, min_delta=0.001)
    for ep in range(epochs):
        tl, tr, tk = 0.0, 0.0, 0.0
        for x, _, _ in tqdm(train_loader, leave=False):
            x = x.to(device)
            # TODO START
            xhat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #raise NotImplementedError('Implement VAE training step')
            # TODO END
            tl += loss.item() * x.size(0)
            tr += recon.item() * x.size(0)
            tk += kl.item() * x.size(0)

        val_metrics = evaluate_vae(model, val_loader, beta=beta, flatten_input=flatten_input)
        val_loss = val_metrics['loss']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        n = len(train_loader.dataset)
        hist.append({
            'train_loss': tl/n,
            'train_recon_bce': tr/n,
            'train_kl': tk/n,
            'val_loss': val_loss
        })
        print(f'Epoch {ep+1}/{epochs} | Loss: {tl/n:.4f} | Recon: {tr/n:.4f} | KL: {tk/n:.4f}')

        if early_stopper.early_stop(val_loss):
            print(f'Early stopping ativado na época {ep+1} com val_loss={val_loss:.4f}')
            break
    return hist

def evaluate_vae(model, loader, beta=0.7, flatten_input=False):
    model.eval()
    tl, tr, tk, tm, ta, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            xhat, mu, logvar = model(x)
            b = x.size(0)
            loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=beta)
            # TODO START
            tl += loss.item() * b
            tr += recon.item() * b
            tk += kl.item() * b
            tm += F.mse_loss(xhat, x, reduction='sum').item()
            ta += F.l1_loss(xhat, x, reduction='sum').item()
            n += b
            #raise NotImplementedError('Implement VAE evaluation loop')
            # TODO END
    numel = x[0].numel()
    return {'loss': tl/n, 'recon_bce': tr/n, 'kl': tk/n, 'mse': tm/(n*numel), 'mae': ta/(n*numel), 'rmse': math.sqrt(tm / (n * numel))}


def train_condvae(model, train_loader, val_loader, optimizer, save_path, epochs=20, beta=0.7, num_classes=10):
    model.train()
    hist = []
    best_val_loss = float('inf')
    early_stopper = EarlyStopper(patience=10, min_delta=0.001)

    for ep in range(epochs):
        model.train()
        tl, tr, tk = 0.0, 0.0, 0.0

        for images, labels, _ in tqdm(train_loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            c = F.one_hot(labels, num_classes=num_classes).float()

            xhat, mu, logvar = model(images, c)
            loss, recon, kl = vae_loss(xhat, images, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            tl += loss.item() * batch_size
            tr += recon.item() * batch_size
            tk += kl.item() * batch_size

        val_loss = evaluate_condvae(model, val_loader, beta=beta, num_classes=num_classes)['loss']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        n = len(train_loader.dataset)
        hist.append({'train_loss': tl/n, 'train_recon_bce': tr/n, 'train_kl': tk/n, 'val_loss': val_loss})
        print(f'Epoch {ep+1}/{epochs} | train_loss={tl/n:.4f} train_recon={tr/n:.4f} train_kl={tk/n:.4f}')

        if early_stopper.early_stop(val_loss):
            print(f'Early stopping ativado na época {ep+1}')
            break
    return hist


def evaluate_condvae(model, loader, beta=0.7, num_classes=10):
    model.eval()
    tl, tr, tk, tm, ta, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        #for x, _, _ in loader:
            #x = x.to(device)
            #xhat, mu, logvar = model(x)
            #b = x.size(0)
            #loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=beta)

        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            #One-Hot
            c = F.one_hot(labels, num_classes=num_classes).float()
            
            xhat, mu, logvar = model(images, c)
            b = images.size(0)
            loss, recon, kl = vae_loss(xhat, images, mu, logvar, beta=beta)
            
            tl += loss.item() * b
            tr += recon.item() * b
            tk += kl.item() * b
            tm += F.mse_loss(xhat, images, reduction='sum').item()
            ta += F.l1_loss(xhat, images, reduction='sum').item()
            n += b
            
    numel = images[0].numel()
    return {'loss': tl/n, 'recon_bce': tr/n, 'kl': tk/n, 'mse': tm/(n*numel), 'mae': ta/(n*numel)}


def evaluate_condvae_per_class(model, loader, num_classes, beta=0.7):
    model.eval()
    # Dicionário para guardar as métricas de cada classe
    stats = {i: {'loss': 0.0, 'recon': 0.0, 'count': 0} for i in range(num_classes)}
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            c = F.one_hot(labels, num_classes=num_classes).float()
            
            # Forward (passando labels para obter resultados individuais)
            xhat, mu, logvar = model(images, c)
            
            for i in range(images.size(0)):
                label = labels[i].item()
                # Calcula a loss apenas para esta imagem
                x_sample = images[i].unsqueeze(0)
                c_sample = c[i].unsqueeze(0)
                xhat_sample, mu_sample, logvar_sample = model(x_sample, c_sample)
                
                loss, recon, _ = vae_loss(xhat_sample, x_sample, mu_sample, logvar_sample, beta=beta)
                
                stats[label]['loss'] += loss.item()
                stats[label]['recon'] += recon.item()
                stats[label]['count'] += 1
                
    # médias
    for i in range(num_classes):
        if stats[i]['count'] > 0:
            stats[i]['loss'] /= stats[i]['count']
            stats[i]['recon'] /= stats[i]['count']
            
    return stats


def train_vqvae(model, train_loader, val_loader, optimizer, epochs=50, save_path=None):
    #Eliminamos o beta e o cálculo de kl. 
    #Substituímos pelo quant_loss que o seu modelo VQ-VAE agora devolve

    model.train()
    hist = []

    early_stopper = EarlyStopper(patience=20, min_delta=1e-4 )

    best_val_loss = float('inf')
    early_stop_counter = 0

    for ep in range(epochs):
        tl, tr, tq = 0.0, 0.0, 0.0 # tq = total quantization loss 

        for images, _, _ in tqdm(train_loader, leave=False):
            images = images.to(device)
            optimizer.zero_grad()
            
            # Forward
            xhat, quant_loss = model(images)
            # Recon Loss (MSE é o padrão para VQ-VAE)
            recon_loss = F.mse_loss(xhat, images)
            # Loss Total
            loss = recon_loss + quant_loss

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            tl += loss.item() * batch_size
            tr += recon_loss.item() * batch_size
            tq += quant_loss.item() * batch_size

        n = len(train_loader.dataset)
        train_stats = {'train_loss': tl/n, 'train_recon': tr/n, 'train_quant_loss': tq/n}

        val_metrics = evaluate_vqvae(model, val_loader)
        val_loss = val_metrics['loss']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        n = len(train_loader.dataset)
        hist.append({
            'train_loss': tl/n,
            'train_recon_bce': tr/n,
            'train_quant_loss': tq/n,
            'val_loss': val_loss
        })
        print(f'Epoch {ep+1}/{epochs} | Loss: {tl/n:.4f} | Recon: {tr/n:.4f} | Quant: {tq/n:.4f}')

        if early_stopper.early_stop(val_loss):
            print(f'Early stopping ativado na época {ep+1} com val_loss={val_loss:.4f}')
            break

    return hist

#Removemos as métricas de KL e adicionei a métrica de quantização

def evaluate_vqvae(model, loader):
    model.eval()
    tl, tr, tq, tm, ta, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            
            
            xhat, quant_loss = model(images)

            recon_loss = F.mse_loss(xhat, images)
            loss = recon_loss + quant_loss
            
            b = images.size(0)
            tl += loss.item() * b
            tr += recon_loss.item() * b
            tq += quant_loss.item() * b
            tm += F.mse_loss(xhat, images, reduction='sum').item()
            ta += F.l1_loss(xhat, images, reduction='sum').item()
            n += b
            
    numel = images[0].numel()
    return {
        'loss': tl/n, 
        'recon_loss': tr/n, 
        'quant_loss': tq/n, 
        'mse': tm/(n*numel), 
        'mae': ta/(n*numel),
        'rmse': math.sqrt(tm / (n * numel)) 
    }

def train_vae_denoising(model, train_loader, val_loader, optimizer, save_path=None,
                        epochs=20, beta=0.7, noise_factor=0.3, flatten_input=False):
    early_stopper = EarlyStopper(patience=10, min_delta=0.0)
    model.train()
    hist = []
    best_val_loss = float('inf')

    for ep in range(epochs):
        model.train()
        tl, tr, tk = 0.0, 0.0, 0.0
        for x, _, _ in tqdm(train_loader, leave=False, desc=f"Epoch {ep+1} Training"):
            x = x.to(device)
            x_noisy = torch.clamp(x + torch.randn_like(x) * noise_factor, 0., 1.)

            xhat, mu, logvar = model(x_noisy)
            #loss, recon, kl = vae_loss(xhat, x, mu, logvar, beta=beta)
            loss, recon, kl = vae_loss_mse(xhat, x, mu, logvar, beta=beta)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tl += loss.item() * x.size(0)
            tr += recon.item() * x.size(0)
            tk += kl.item() * x.size(0)

        val_metrics = evaluate_vae(model, val_loader, beta=beta, flatten_input=flatten_input)
        val_loss = val_metrics['loss']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        n = len(train_loader.dataset)
        hist.append({
            'train_loss': tl/n,
            'train_recon_bce': tr/n,
            'train_kl': tk/n,
            'val_loss': val_loss
        })
        print(f'Epoch {ep+1}/{epochs} | Loss: {tl/n:.4f} | Recon: {tr/n:.4f} | KL: {tk/n:.4f}')

        if early_stopper.early_stop(val_loss):
            print(f'Early stopping ativado na época {ep+1} com val_loss={val_loss:.4f}')
            break
    return hist

def fit_or_load_model(model, run_name, train_fn, load_if_available=True, **train_kwargs):
    model_path = Path(f"{run_name}.pth")
    history_path = Path(f"{run_name}_history.json")

    if load_if_available and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        with open(history_path, 'r') as f:
            history = json.load(f)
        print(f'Modelo carregado de: {model_path}')
        return history


    history = train_fn(model=model, save_path=str(model_path), **train_kwargs)

    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    print(f'Treino concluído. Melhor modelo guardado em: {model_path}')
    return history

