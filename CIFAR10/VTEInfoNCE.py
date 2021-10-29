import torch
import torch.nn as nn
import pytorch_lightning as pl

from BasicBlock import *

class VTEInfoNCE(pl.LightningModule):
    def __init__(self, in_channel, learning_rate, extractor, latent_dim=512):
        super(VTEInfoNCE, self).__init__()

        self.in_channel = in_channel
        self.latent_dim = latent_dim
        self.extractor = extractor
        self.extractor.freeze()

        CHANNEL1 = 192
        CHANNEL2 = 160
        CHANNEL3 = 96
        CHANNEL4 = 64
        CHANNEL5 = 8

        #encoder
        self.block1 = nn.Sequential(
                    BasicBlock(in_channel, CHANNEL1, 5),
                    BasicBlock(CHANNEL1, CHANNEL2, 1),
                    BasicBlock(CHANNEL2, CHANNEL3, 1),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )

        self.block2 = nn.Sequential(
                    BasicBlock(CHANNEL3, CHANNEL1, 5),
                    BasicBlock(CHANNEL1, CHANNEL1, 1),
                    BasicBlock(CHANNEL1, CHANNEL5, 1),
                    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Flatten()
                )


        self.mu_encoder = nn.Linear(self.latent_dim, self.latent_dim)
        self.logsigma_encoder = nn.Linear(self.latent_dim, self.latent_dim)


        self.linear1 = nn.Sequential( 
                        nn.Linear(self.latent_dim, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, latent_dim)
                    )

        self.linear2 = nn.Sequential( 
                        nn.Linear(self.latent_dim, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, latent_dim)
                    )

        self.learning_rate = learning_rate


    def forward(self, x):
        infer = self.block1(x)
        infer = self.block2(infer)

        mu = self.mu_encoder(infer)
        logsigma = self.logsigma_encoder(infer)

        z = self.reparameterize(mu, logsigma)

        return mu, logsigma, z

    def training_step(self, batch, batch_idx):
        x = batch
        x1, x2, trns_matrix = x[0], x[1], x[2].view(-1, 8)
        batch_size = x1.size()[0]

        mu_z, logsigma_z, z = self.extractor.forward(x2)

        infer_zhat = self.block1(x1)
        infer_zhat = self.block2(infer_zhat)

        mu_zhat = self.mu_encoder(infer_zhat)
        logsigma_zhat = self.logsigma_encoder(infer_zhat)

        zhat = self.reparameterize(mu_zhat, logsigma_zhat)

        f_zhat = self.linear1(zhat)
        f_z = self.linear2(z)

        samples_f_z = torch.repeat_interleave(f_z, repeats=batch_size, dim=0)
        samples_f_zhat = f_zhat.repeat(repeats=(batch_size, 1))

        product = torch.sum(samples_f_z * samples_f_zhat, 1).view(-1, 1)

        loss = None
        for i in range(batch_size):
            positive_pair = product[i*batch_size + i]
            negative_pairs = product[i*batch_size: i*batch_size + batch_size].view(-1, 1)
            max_pair = torch.max(negative_pairs)
            sum_exp = torch.sum(torch.exp(negative_pairs - max_pair), 0).view(-1, 1)
            log_sum_exp = max_pair + torch.log(sum_exp)

            sample_loss = positive_pair - log_sum_exp

            if loss == None:
                loss = sample_loss
            else:
                loss = torch.cat((loss, sample_loss), 0)

        loss = torch.mean(loss)
        
        self.log('train_loss', loss)
        return loss

    def reparameterize(self, mu, logsigma):
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)