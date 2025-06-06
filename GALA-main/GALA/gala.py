import random, os
import torch.autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from wgmnn import order_selection, generate_target_dataset
from wgmnn_grokkkkk import modified_rwMNN
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scanpy as sc
from diffusion import Diffusion
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###########################
# 1. MinibatchStdLayer
###########################
class MinibatchStdLayer(nn.Module):
    def __init__(self, group_size=4, num_channels=1):
        super(MinibatchStdLayer, self).__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C = x.size()
        G = min(self.group_size, N)
        if N % G != 0:
            pad_size = G - (N % G)
            x = torch.cat([x, x[:pad_size]], dim=0)
            N = x.size(0)
        group_size = N // G
        y = x.view(G, group_size, C)
        y = y - y.mean(dim=1, keepdim=True)
        y = y.pow(2).mean(dim=1)
        y = torch.sqrt(y + 1e-8)
        y = y.mean(dim=1, keepdim=True)
        y = y.repeat_interleave(group_size, dim=0)
        x = torch.cat([x, y], dim=1)
        return x


###########################
# 2. DiscriminatorBlock
###########################
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Mish()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Mish()
        )
        self.residual = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Mish()
        )

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out


###########################
# 3. Discriminator
###########################
class discriminator(nn.Module):
    def __init__(self, data_size, time_embedding_dim=16, timesteps=1000):
        super(discriminator, self).__init__()
        self.time_embedding = nn.Embedding(num_embeddings=timesteps, embedding_dim=time_embedding_dim)
        self.block1 = DiscriminatorBlock(data_size + time_embedding_dim, 128)
        self.mbstd = MinibatchStdLayer()
        self.block2 = DiscriminatorBlock(128 + 1, 128)
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, 1)
        )

    def forward(self, x, t):
        if isinstance(t, int):
            t = torch.full((x.size(0),), t, dtype=torch.long, device=x.device)

        t_emb = self.time_embedding(t)
        t_emb = t_emb.view(t_emb.size(0), -1)
        x = torch.cat([x, t_emb], dim=1)
        x = self.block1(x)
        x = self.mbstd(x)
        x = self.block2(x)
        validity = self.fc(x)
        return validity


###########################
# 4. Encoder
###########################
class Encoder(nn.Module):
    def __init__(self, input_dim=2000, latent_dim=256):
        super(Encoder, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Dropout(0.1)
        )
        self.enc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.1)
        )
        self.enc3 = nn.Linear(512, latent_dim * 2)  

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        z_params = self.enc3(x2)
        z_mean = z_params[:, :256]
        z_log_var = z_params[:, 256:]
        eps = torch.randn_like(z_mean)
        z = z_mean + torch.exp(0.5 * z_log_var) * eps
        return z, z_mean, z_log_var


###########################
# 5. Decoder
###########################
class Decoder(nn.Module):
    def __init__(self, latent_dim=256, output_dim=2000):
        super(Decoder, self).__init__()
        self.dec4 = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.Mish()
        )
        self.dec5 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.Mish()
        )
        self.dec6 = nn.Linear(1024, output_dim)

    def forward(self, z):
        x4 = self.dec4(z)
        x5 = self.dec5(x4)
        output = self.dec6(x5)
        return output


class Generator(nn.Module):
    def __init__(self, input_dim=2000, latent_dim=256):
        super(Generator, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def latent_from_encoder(self, x):
        z, _, _ = self.encoder(x)
        return z

    def inference(self, x):
        z, z_mean, z_log_var = self.encoder(x)
        mu = self.decoder(z)
        return z, z_mean, z_log_var, mu

    def forward(self, x):
        z, z_mean, z_log_var, output = self.inference(x)
        loss_mse = F.mse_loss(output, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return z, loss_mse, kl_loss


def initialize_models(input_dim=2000, latent_dim=256):
    Dis = discriminator(data_size=latent_dim).to(device)
    AE = Generator(input_dim, latent_dim).to(device)
    return Dis, AE


def calculate_gradient_penalty(real_data, fake_data, D, t, center=1, p=2):
    device = real_data.device
    alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha = alpha.expand_as(real_data)  # [N, latent_dim]
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)

    prob_interpolated = D(interpolated, t)
    grad = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated, device=device),
        create_graph=True,
        retain_graph=True
    )[0]

    grad_penalty = ((grad.norm(2, dim=1) - center) ** p).mean()
    return grad_penalty


###########################
# 9. 训练函数 (带稀疏项)
###########################
def train(label_data, train_data, query_data, config):
    Dis, AE = initialize_models(input_dim=label_data.shape[1], latent_dim=256)

    dis_optimizer = torch.optim.AdamW(Dis.parameters(), lr=config['lr'], betas=(config['b1'], config['b2']))
    ae_optimizer = torch.optim.AdamW(AE.parameters(), lr=config['lr'], betas=(config['b1'], config['b2']))
    ec_optimizer = torch.optim.AdamW(AE.encoder.parameters(), lr=config['lr'], betas=(config['b1'], config['b2']))

    scheduler_dis = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(dis_optimizer, T_0=1, T_mult=2, eta_min=1e-6)
    scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(ae_optimizer, T_0=2, T_mult=2, eta_min=1e-6)
    scheduler_ec = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(ec_optimizer, T_0=1, T_mult=2, eta_min=1e-6)

    diffusion = Diffusion(beta_schedule='linear', t_min=10, t_max=1000)
    diffusion.set_diffusion_process(1000, beta_schedule='linear')

    Dis.train()
    AE.train()

    d_losses = []
    g_losses = []

    print('Start adversarial training...')


    for epoch in range(config['epoch']):
        sample_index = np.random.choice(len(label_data), size=min(10 * 1024, len(label_data)), replace=False)
        label_data_sample = torch.FloatTensor(label_data[sample_index]).to(device)
        train_data_sample = torch.FloatTensor(train_data[sample_index]).to(device)

        training_set = data_utils.TensorDataset(train_data_sample, label_data_sample)
        dataloader = DataLoader(training_set, batch_size=config['batch_size'])

        for i, (false_data, true_data) in enumerate(dataloader): 
            true_data = true_data.to(device)
            false_data = false_data.to(device)

            ae_optimizer.zero_grad()
            z_true, loss_mse_true, kl_loss_true = AE(true_data)
            sparsity_loss_true = z_true.abs().mean()
            reconst_loss_true = (config['lambda_mse'] * loss_mse_true
                                 + config['lambda_kl'] * kl_loss_true
                                 + config['lambda_sparse'] * sparsity_loss_true)

            reconst_loss_true.backward()
            ae_optimizer.step()

            ec_optimizer.zero_grad()

            z_false, loss_mse_false, kl_loss_false = AE(false_data)
            sparsity_loss_false = z_false.abs().mean()
            enc_loss_false = config['lambda_kl'] * kl_loss_false + config['lambda_sparse'] * sparsity_loss_false
            enc_loss_false.backward()
            ec_optimizer.step()

            dis_optimizer.zero_grad()

            z_true, _, _, _ = AE.inference(true_data)
            real_diffused, t_real = diffusion(z_true)  
            real_out = Dis(real_diffused, t_real)
            real_loss = -torch.mean(real_out)

            z_false, _, _, _ = AE.inference(false_data)
            fake_diffused, t_fake = diffusion(z_false)
            fake_out = Dis(fake_diffused, t_fake)
            fake_loss = torch.mean(fake_out)

            div = calculate_gradient_penalty(real_diffused, fake_diffused, Dis, t_fake, center=1)
            D_loss = real_loss + fake_loss + config['lambda_1'] * div
            D_loss.backward()
            dis_optimizer.step()


            if i % config['n_critic'] == 0:
                z_false, _, _, _ = AE.inference(false_data)
                fake_diffused, t_fake = diffusion(z_false)
                fake_out = Dis(fake_diffused, t_fake)
                G_loss = -torch.mean(fake_out)

                ec_optimizer.zero_grad()
                G_loss.backward()
                ec_optimizer.step()

        scheduler_dis.step()
        scheduler_ae.step()
        scheduler_ec.step()

        d_losses.append(D_loss.item())
        g_losses.append(G_loss.item())

        if epoch % 50 == 0:
            print(f"This is epoch: {epoch}")
            print(f"d step loss: {D_loss.item()}")
            print(f"g step loss: {G_loss.item()}")

    print("Train step finished.")
    print("Generate corrected data...")

    AE.eval()
    test_data = torch.FloatTensor(query_data).to(device)
    with torch.no_grad():
        _, _, _, mu = AE.inference(test_data)

    test_list = mu.detach().cpu().numpy()

    return test_list, AE, d_losses, g_losses


def run_gala(adata, config, batch_key='batch', order=None):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    os.environ['PYTHONHASHEED'] = str(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True

    if not order:
        order = order_selection(adata, key=batch_key)

    print('The sequential mapping order will be: ', '->'.join(map(str, order)))

    adata = generate_target_dataset(adata, order)
    ref_data_ori = adata[adata.obs['batch'] == order[0]].X

    for bat in order[1:]:
        print(f"########################## Mapping {bat} to the reference data #####################")
        batch_data_ori = adata[adata.obs['batch'] == bat].X

        label_data, train_data, ref_indices, query_indices, init_pairs = modified_rwMNN(
            ref_data_ori, batch_data_ori,
            k_intra=None, k_cross=None, sigma=None,
            walk_steps=50, filtering=False, k_filter=10,
            metric='euclidean', reduction='pca', norm=True
        )
        print("######################## Finish pair finding ########################")
        if 'cell_type' in adata.obs.columns:
            ref_init_indices, query_init_indices = init_pairs
            ref_cells = adata[adata.obs['batch'] == order[0]].obs_names[ref_init_indices]
            query_cells = adata[adata.obs['batch'] == bat].obs_names[query_init_indices]
            ref_labels = adata[ref_cells].obs['cell_type']
            query_labels = adata[query_cells].obs['cell_type']
            all_categories = pd.unique(pd.concat([ref_labels, query_labels]))

            ref_labels = ref_labels.astype('category').cat.set_categories(all_categories)
            query_labels = query_labels.astype('category').cat.set_categories(all_categories)

            accuracy = np.mean(ref_labels.values == query_labels.values)

            print(f"MNN对的匹配准确度: {accuracy:.6f}")

        if 'cell_type' in adata.obs.columns:
            ref_cells = adata[adata.obs['batch'] == order[0]].obs_names[ref_indices]
            query_cells = adata[adata.obs['batch'] == bat].obs_names[query_indices]
            ref_labels = adata[ref_cells].obs['cell_type']
            query_labels = adata[query_cells].obs['cell_type']
            all_categories = pd.unique(pd.concat([ref_labels, query_labels]))

            ref_labels = ref_labels.astype('category').cat.set_categories(all_categories)
            query_labels = query_labels.astype('category').cat.set_categories(all_categories)

            accuracy = np.mean(ref_labels.values == query_labels.values)

            print(f"批次 {bat} 的匹配准确度: {accuracy:.6f}")

        remove_batch_data, G_tar, d_losses, g_losses = train(label_data, train_data, batch_data_ori, config)

        ref_data_ori = np.vstack([ref_data_ori, remove_batch_data])

    print("######################## Finish all batch correction ########################")

    adata_new = sc.AnnData(ref_data_ori)
    adata_new.obs['batch'] = list(adata.obs['batch'])
    if adata.obs.columns.str.contains('celltype').any():
        adata_new.obs['celltype'] = list(adata.obs['celltype'])
    adata_new.var_names = adata.var_names
    adata_new.var_names.name = 'Gene'
    adata_new.obs_names = adata.obs_names
    adata_new.obs_names.name = 'CellID'

    return adata_new


def plot_losses(d_losses, g_losses, save_path='None'):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    if save_path and save_path != 'None':
        plt.savefig(save_path)
        print(f"Plot saved at {save_path}")



def calculate_gradient_penalty(real_data, fake_data, D, t, center=1, p=2):
    device = real_data.device
    alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha = alpha.expand_as(real_data)  # [N, latent_dim]

    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)   

    prob_interpolated = D(interpolated, t)

    grad = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated, device=device),
        create_graph=True,
        retain_graph=True
    )[0]

    grad_penalty = ((grad.norm(2, dim=1) - center) ** p).mean()
    return grad_penalty





