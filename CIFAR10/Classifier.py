import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import accuracy

class CLassifier(pl.LightningModule):
    def __init__(self, input_dim=512, n_class=10, extractor, lr=0.05):
        super(Classifier, self).__init__()
        
        n_feat = min(n_class*20, 2048)
        self.learning_rate = lr

        self.classifier = nn.Sequential(
                nn.Linear(input_dim, n_feat, bias=False),
                nn.BatchNorm1d(n_feat),
                nn.ReLU(inplace=True),
                nn.Linear(n_feat, n_feat, bias=False),
                nn.BatchNorm1d(n_feat),
                nn.ReLU(inplace=True),
                nn.Linear(n_feat, n_class)
            )

        self.extractor = extractor
        self.extractor.freeze()
        self.save_hyperparameters()
        

    def forward(self, x):
        mu, logsigma, z = self.extractor.forward(x)
        logits = self.classifier(mu)
        return nn.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, logsigma, z = self.extractor.forward(x)
        logits = self(mu)
        loss = nn.CrossEntropyLoss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        mu, logsigma, z = self.extractor.forward(x)
        logits = self(mu)
        loss = nn.CrossEntropyLoss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

            return {f"{stage}_loss" : loss, f"{stage}_acc" : acc}

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)