import torch
import torch.nn as nn
import pytorch_lightning as pl

class Transformation(pl.LightningModule):
    def __init__(self, in_channel, learning_rate, latent_dim=512):
        super(Transformation, self).__init__()

        TARGET_DIM = 8
        self.in_channel = in_channel
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )

        self.mu_encoder = nn.Linear(2*2*256, self.latent_dim)
        self.logsigma_encoder = nn.Linear(2*2*256, self.latent_dim)

        self.linear = nn.Sequential(
                nn.Linear(self.latent_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, TARGET_DIM)
            )

        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x):
        
        infer = self.conv(x)
    
        mu = self.mu_encoder(infer)
        logsigma = self.logsigma_encoder(infer)

        z = self.reparameterize(mu, logsigma)

        return mu, logsigma, z

    def training_step(self, batch, batch_idx):
        x = batch
        x1, x2, trns_matrix = x[0], x[1], x[2].view(-1, 8)

        infer_z = self.conv(x2)
        
        mu_z = self.mu_encoder(infer_z)
        logsigma_z = self.logsigma_encoder(infer_z)

        z = self.reparameterize(mu_z, logsigma_z)

        decoded_t = self.linear(z)

        criterion = nn.MSELoss() 
        reconstruction_t = criterion(decoded_t, trns_matrix)
        
        self.log('train_loss', reconstruction_t)
        return reconstruction_t

    def reparameterize(self, mu, logsigma):
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
