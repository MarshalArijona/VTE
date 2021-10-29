import torch
import torch.nn as nn
import pytorch_lightning as pl

class VTEBA(pl.LightningModule):
    def __init__(self, in_channel, learning_rate, extractor, latent_dim=512):
        super(VTEBA, self).__init__()

        self.in_channel = in_channel
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        
        self.extractor = extractor
        self.extractor.freeze()

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
                nn.Linear(1024, self.latent_dim)
        )

        self.learning_rate = learning_rate

    def forward(self, x):
        infer = self.conv(x)

        mu = self.mu_encoder(infer)
        logsigma = self.logsigma_encoder(infer)

        z = self.reparameterize(mu, logsigma)

        return mu, logsigma, z

    def training_step(self, batch, batch_idx):
        x = batch
        x1, x2, trns_matrix = x[0], x[1], x[2].view(-1, 8)
        batch_size = x1.size()[0]

        mu_z, logsigma_z, z = self.extractor.forward(x2)

        infer_zhat = self.conv(x1)

        mu_zhat = self.mu_encoder(infer_zhat)
        logsigma_zhat = self.logsigma_encoder(infer_zhat)

        zhat = self.reparameterize(mu_zhat, logsigma_zhat)

        infer_z = self.linear(zhat) 

        criterion = nn.MSELoss() 
        reconstruction_z = criterion(infer_z, z)
        
        self.log('train_loss', reconstruction_z)
        return reconstruction_z

    def reparameterize(self, mu, logsigma):
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
