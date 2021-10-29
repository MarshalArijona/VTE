import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

class Classifier(pl.LightningModule):
    def __init__(self, extractor, input_dim=512, n_class=200, lr=0.05):
        super(Classifier, self).__init__()
        
        n_feat = 2048
        self.input_dim = input_dim
        self.learning_rate = lr
        self.n_class = n_class
        
        self.classifier = nn.Sequential(
                #nn.Conv2d(2, 96, kernel_size=3, padding=1),
                #nn.BatchNorm2d(96),
                #nn.ReLU(),
                #nn.MaxPool2d(kernel_size=3, stride=2),
                #nn.Flatten(),
                nn.Linear(self.input_dim, n_feat, bias=False),
                #nn.Linear(96*7*7, n_feat, bias=False),
                nn.BatchNorm1d(n_feat),
                nn.ReLU(inplace=True),
                nn.Linear(n_feat, n_feat, bias=False),
                nn.BatchNorm1d(n_feat),
                nn.ReLU(inplace=True),
                nn.Linear(n_feat, self.n_class)
            )

        self.extractor = extractor
        self.extractor.freeze()
        #self.save_hyperparameters()
        
    def forward(self, x):
        mu, logsigma, z = self.extractor.forward(x)
        #z = z.view(-1, 2, 16, 16)
        logits = self.classifier(z)
        return nn.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        #sub = x[:20]
        #subx2 = x2[:10]
        #sub = torch.cat((subx1, subx2), 0)
        #grid_img = torchvision.utils.make_grid(sub, nrow=20)
        #plt.imsave("samples2.png", grid_img.permute(1, 2, 0).detach().cpu().numpy())
        
        mu, logsigma, z = self.extractor.forward(x)
        #z = z.view(-1, 2, 16, 16)
        logits = self.classifier(z)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        mu, logsigma, z = self.extractor.forward(x)
        #z = z.view(-1, 2, 16, 16)
        logits = self.classifier(z)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
