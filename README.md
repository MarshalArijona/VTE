# VTE

# Abstract
The success of deep learning on computer vision tasks is due to the convolution layer that equivaries to thetranslation transformation. Several works attempt to extend the notion of equivariance into more general trans-formations. Autoencoding variational transformation (AVT) achieves state of art by approaching the problemfrom the information theory perspective.  The model involves the computation of mutual information, whichleads to a more general transformation equivariant representation model.  In this research, we investigate thealternatives of AVT called variational transformation equivariant (VTE). We utilize the Barber-Agakov andInfo-NCE mutual information estimation to optimize VTE. Furthermore, we also propose a sequential mecha-nism to train our VTE. Results of experiments demonstrate that VTE outperforms AVT on image classificationtasks

# Method
## Predictive Transformation

## VTE Barber-Agakov

# Require
1. Python == 3.7
2. PyTorch == 1.9
3. PyTorch-Lightning == 1.5
4. Torchvsion == 0.11.0

# Disclaimer
Some of our codes reuse the github project [AVT](https://github.com/maple-research-lab/AVT-pytorch/blob/master/)
