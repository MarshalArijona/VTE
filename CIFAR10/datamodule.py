from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from dataset import CIFAR10

import pytorch_lightning as pl
import PIL

class DataModuleCIFAR10Transformation(pl.LightningDataModule):
	def __init__(self, batch_size, download_dir, scale=(0.8, 1.2), shift=4.0):
		super().__init__()
		self.download_dir = download_dir
		self.batch_size = batch_size

		#all three of the transformation operations and parameters adapt Qi, et al., 2019
		self.matrix_transform=transforms.Compose([
                               transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                           ])
		
		self.transform_pre=transforms.Compose([
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                           ])
		
		self.transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ])


		self.fillcolor = (128, 128, 128)
		self.resample = PIL.Image.BILINEAR
		self.scale = scale
		self.shift = shift


	def prepare_data(self):
		
		CIFAR10(root=self.download_dir,  shift=self.shift, scale=self.scale,  
							   download=True, train=True, resample=self.resample, matrix_transform=self.matrix_transform,
							   transform_pre=self.transform_pre, transform=self.transform)
		
		CIFAR10(root=self.download_dir, shift=self.shift, scale=self.scale,
									  download=True, train=False, resample=self.resample, matrix_transform=self.matrix_transform,
									  transform_pre=self.transform_pre, transform=self.transform)

	def setup(self, stage=None): 
		if stage == "fit" or stage is None:
			
			self.train_cifar = CIFAR10(root=self.download_dir,  shift=self.shift, scale=self.scale,  
							   download=True, train=True, resample=self.resample, matrix_transform=self.matrix_transform,
							   transform_pre=self.transform_pre, transform=self.transform)

		if stage == "test" or stage is None:
			
			self.test_cifar = CIFAR10(root=self.download_dir, shift=self.shift, scale=self.scale,
									  download=True, train=False, resample=self.resample, matrix_transform=self.matrix_transform,
									  transform_pre=self.transform_pre, transform=self.transform)


	def train_dataloader(self):
		return DataLoader(self.train_cifar, batch_size=self.batch_size, shuffle=True)

	def test_dataloader(self):
		return DataLoader(self.test_cifar, batch_size=self.batch_size, shuffle=False)


class DataModuleCIFAR10(pl.LightningDataModule):
	def __init__(self, download_dir, batch_size=256, n_train=50000, split=0.1):
		super().__init__()
		self.download_dir = download_dir
		self.batch_size = batch_size

		#all three of the transformation operations and parameters adapt Qi, et al., 2019
		self.transform_train = transforms.Compose([
        							transforms.RandomCrop(32, padding=4),
        							transforms.RandomHorizontalFlip(),
        							transforms.ToTensor(),
        							transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    							])

		self.transform_test = transforms.Compose([
        							transforms.ToTensor(),
        							transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    							])

		self.split = split
		self.n_train = n_train

	def prepare_data(self):
		
		datasets.CIFAR10(root=self.download_dir, download=True, train=True, transforms=self.transform_train)
		
		datasets.CIFAR10(root=self.download_dir, download=True, train=False, trasnforms=self.transform_test)

	def setup(self, stage=None):
		SIZE = 50000
		SEED = 42

		if stage == "fit" or stage is None:
			cifar = datasets.CIFAR10(root=self.download_dir, download=True, train=True, transforms=self.transform_train)
			
			valid_size = int(self.split * self.n_train)
			train_size = self.n_train - valid_size
			residual_size = SIZE - self.n_train 

			generator = torch.Generator().manual_seed(SEED)
			self.valid_cifar, self.train_cifar, residual = random_split(cifar, [valid_size, train_size, residual_size], generator=generator)
			
		
		if stage == "test" or stage is None:
			self.test_cifar = datasets.CIFAR10(root=self.download_dir, download=True, train=False, trasnforms=self.transform_test)

	def train_dataloader(self):
		return DataLoader(self.train_cifar, batch_size=self.batch_size, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.valid_cifar, batch_size=self.batch_size, shuffle=False)

	def test_dataloader(self):
		return DataLoader(self.test_cifar, batch_size=self.batch_size, shuffle=False)

