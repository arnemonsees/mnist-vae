#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch
#
import torch.nn as nn
import torch.optim as optim
#
from sklearn.decomposition import PCA
#
import mnist_io

if torch.cuda.is_available():  
  dev = 'cuda:0'
else:  
  dev = 'cpu'

# based on 2014__Kingma__Auto-Encoding_Variational_Bayes
class Encoder(nn.Module): # Gaussian output
    def __init__(self, dim_img2, dim_inbetween, dim_z):
        super(Encoder, self).__init__()
        self.fc3 = nn.Linear(dim_img2, dim_inbetween, bias=True, device=dev, dtype=torch.double)
        self.fc4 = nn.Linear(dim_inbetween, dim_z, bias=True, device=dev, dtype=torch.double)
        self.fc5 = nn.Linear(dim_inbetween, dim_z, bias=True, device=dev, dtype=torch.double)

    def forward(self, x):
        f3_out = self.fc3(torch.flatten(x, start_dim=1))
        h = torch.tanh(f3_out)
        mu = self.fc4(h)
        sigma = torch.sqrt(torch.exp(self.fc5(h)))
        return mu, sigma

# based on 2014__Kingma__Auto-Encoding_Variational_Bayes
class Decoder(nn.Module): # Bernoulli output % IMPROVE: use continous Bernoulli (only relevant when cross entropy loss is used?)
    def __init__(self, dim_z, dim_inbetween, dim_img):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(dim_z, dim_inbetween, bias=True, device=dev, dtype=torch.double)
        self.fc2 = nn.Linear(dim_inbetween, dim_img**2, bias=True, device=dev, dtype=torch.double)
        
    def forward(self, z):
        y = torch.sigmoid(self.fc2(torch.tanh(self.fc1(z)))).reshape(z.shape[0], dim_img, dim_img)
        return y

class VAE(nn.Module):
    def __init__(self, dim_img, dim_z, dim_inbetween_encoder, dim_inbetween_decoder):
        super(VAE, self).__init__()
        self.encoder = Encoder(dim_img**2, dim_inbetween_encoder, dim_z)
        self.decoder = Decoder(dim_z, dim_inbetween_decoder, dim_img)
        #
        self.samples = torch.zeros(dim_z, device=dev)
    
    def encode(self, img):
      mu, sigma = self.encoder(img)
      return mu, sigma

    def decode(self, z):
      y = self.decoder(z)
      return y

    def forward(self, img):
        # encode
        mu, sigma = self.encode(img)
        # sampling
        torch.randn(dim_z, out=self.samples) # CHECK: should I rather sample size_batch*dim_z random numbers?
        z = mu + sigma * self.samples[None, :]
        # decode
        y = self.decode(z)
        return mu, sigma, y

def vae_loss(x, mu, sigma, y):
#     log_p_of_x_given_z = torch.sum(x * torch.log(y) + (1.0 - x) * torch.log(1.0 - y)) # aka cross entropy loss
    l2 = torch.sum((y - x)**2) # aka L2 loss
    reconstruction_loss = l2 # use either, see https://stats.stackexchange.com/questions/350211/loss-function-autoencoder-vs-variational-autoencoder-or-mse-loss-vs-binary-cross
    # KL divergence is always positive!
    # use it to enforce that the latent states the encoder generates are actually drawn from a Gaussian distribution
    # i.e. calculate KL divergence "between" the distribution of z and N(0,1)
    # only use this simple expression if samples and z are drawn from a Gaussian distribution (paper: see appendix B)
    KL_divergence = -0.5 * torch.sum(1.0 + torch.log(sigma**2) - mu**2 - sigma**2) 
    loss = reconstruction_loss + KL_divergence
    return loss    
    
def plot_imgs(x_plot, y_plot):
  nImgs = 5
  fig1 = plt.figure(1, figsize=(20, 4))
  fig1.clear()
  ax = list()
  for i in range(2*nImgs):
    ax.append(fig1.add_subplot(2,nImgs,i+1))
    ax[-1].clear()
    ax[-1].axis('off')
  for i in range(nImgs):
      h_in = ax[i].imshow(x_plot[i], cmap='gray', vmin=0.0, vmax=1.0)
      h_out = ax[i+nImgs].imshow(y_plot[i], cmap='gray', vmin=0.0, vmax=1.0)
  fig1.canvas.draw()
  plt.show(block=False)
  #plt.pause(0.1)
  return

if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True) # for bug fixing

    # using 32 instead of 100
    # see: https://datascience.stackexchange.com/questions/16807/why-mini-batch-size-is-better-than-one-single-batch-with-all-training-data
    # LeCun on Twitter: "Friends dont let friends use minibatches larger than 32." (https://twitter.com/ylecun/status/989610208497360896?lang=en)
    # see paper: https://arxiv.org/abs/1804.07612
    size_batch = 32 # in the paper 100 was used
    train_batch = float(mnist_io.nImgs_train) / float(size_batch)

    imgs = mnist_io.load_images(mnist_io.file_imgs_train, mnist_io.nImgs_train)
    imgs = torch.from_numpy(np.copy(imgs)).type(torch.DoubleTensor).to(dev) # make pytorch tensor and put on device (aka GPU)
    # IMPROVE: rather normalize such that: mean=0, variance=1
    imgs = imgs / 255.0 # normalize such that image intensity values are in [0, 1]
    
    dim_img = mnist_io.img_size # height & width of a single MNIST image
    dim_z = 10 # variable, at least 2 ("ideally" this could be the numbers 0-9, i.e. maybe use CNN as a encoder for classification) (Kingma et al., 2014: see Fig. 5)
    dim_inbetween_encoder = int(500) # in the paper 500 was used for MNIST
    dim_inbetween_decoder = int(500) # in the paper 500 was used for MNIST
    vae = VAE(dim_img, dim_z, dim_inbetween_encoder, dim_inbetween_decoder).to(dev)
    #
    vae_params_list = list(vae.parameters())

    # print network architecture
    print('Network architecture:')
    print(vae)
    print('Number of parameters:')
    print('\t{:01d}'.format(len(vae_params_list)))
    print('Parameter dimensions:')
    for i in range(len(vae_params_list)):
        print('\t{:02d}:\t'.format(i+1)+'x'.join([str(j) for j in vae_params_list[i].size()]))
    print()

    optimizer = optim.Adagrad(vae.parameters(),
                              lr=0.01,
                              lr_decay=0,
                              weight_decay=0,
                              initial_accumulator_value=0,
                              eps=1e-10,
                              foreach=None,
                              maximize=False) # Kingma et al., 2014 use SGD / Adagrad
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_batch*2.5e1, 
                                                gamma=1e-1) # this is not based on the Kingma et al., 2014 but an individual choice

    # pseudo-implementation of epochs
    print('Training VAE:')
    print()
    nIter = int(train_batch * 5e1)
    #learning_rate = 1e-4 # for simple SGD optimization
    for i in range(nIter):
        img_ids = torch.randint(low=0,
                                high=mnist_io.nImgs_train,
                                size=(size_batch,),
                                dtype=torch.int64,
                                layout=torch.strided,
                                device=dev,
                                requires_grad=False)
        x = imgs[img_ids]
        mu, sigma, y = vae(x)
        #
        #loss = train_batch * torch.sum(vae_loss(x, mu, sigma, y)) # this is the original loss used by Kingma et al., 2014
        loss = torch.sum(vae_loss(x, mu, sigma, y))

        #vae.zero_grad()
        loss.backward()

        # # for simple SGD optimization
        # # decrease learning rate
        # if ((i > 0) and (i % int(train_batch * 1e2) == 0)):
        #     learning_rate = learning_rate * 0.5
        #
        # for f in vae.parameters():
        #     f.data.sub_(f.grad.data * learning_rate)
        
        optimizer.step() # "walk" in direction of negative gradient
        optimizer.zero_grad() # zero gradients for next iteration
        scheduler.step() # adjust learning rate

        # PLOT
        if (i % int(train_batch * 1e1) == 0):
            print('iteration:\t{:07d} / {:07d} '.format(i, nIter))
            print('loss:\t\t\t{:0.2e}'.format(float(loss.data)))
            #print('lr:\t\t\t{:0.2e}'.format(learning_rate)) # for simple SGD optimization
            print('lr:\t\t\t{:0.2e}'.format(scheduler.get_last_lr()[0]))
            print('train size:\t{:07d}'.format(mnist_io.nImgs_train))
            print('batch size:\t{:07d}'.format(size_batch))
            print('ratio:\t\t\t{:07d}'.format(int(train_batch)))
            print()
            x = imgs[:5]
            _, _, y = vae(x)
            x_plot = x.detach().cpu().numpy().astype(np.float64)
            y_plot = y.detach().cpu().numpy().astype(np.float64)
            plot_imgs(x_plot, y_plot)
    
    # PLOT
    print('final:')
    x = imgs[:5]
    _, _, y = vae(x)
    x_plot = x.detach().cpu().numpy().astype(np.float64)
    y_plot = y.detach().cpu().numpy().astype(np.float64)
    plot_imgs(x_plot, y_plot)

    # SAVE
    torch.save(vae.state_dict(), 'weights.pt')
    #
    print('Finished training VAE')
    print()

    ##### ANALYSIS ######
    print('Analyzing VAE:')
    print()
    # SETUP
    labels = mnist_io.load_labels(mnist_io.file_labels_train,
                                  mnist_io.nImgs_train)
    z_mu, z_sigma = vae.encode(imgs)
    z_mu = z_mu.detach().cpu().numpy()
    z_sigma = z_sigma.detach().cpu().numpy()

    # PCA
    pca_mu = PCA(n_components=2,
                 copy=True,
                 whiten=False,
                 svd_solver='auto',
                 tol=0.0,
                 iterated_power='auto')
    pca_sigma = PCA(n_components=2,
                    copy=True,
                    whiten=False,
                    svd_solver='auto',
                    tol=0.0,
                    iterated_power='auto')
    z_mu_pca = pca_mu.fit_transform(z_mu)
    z_sigma_pca = pca_sigma.fit_transform(z_sigma)

    print('Variance explained by PCs:')
    print('\tmu:\tPC1={:0.2f}, PC2={:0.2f}'.format(
        pca_mu.explained_variance_ratio_[0],
        pca_mu.explained_variance_ratio_[1]))
    print('\tsigma:\tPC1={:0.2f}, PC2={:0.2f}'.format(
          pca_sigma.explained_variance_ratio_[0],
          pca_sigma.explained_variance_ratio_[1]))
    print()

    # PLOT
    print('Plotting PCA space:')
    #
    skip_factor = 10
    #
    fig2 = plt.figure(2, figsize=(8, 8))
    fig2.clear()
    ax = list()
    for i in range(4):
      ax.append(fig2.add_subplot(2,2,i+1))
      ax[-1].clear()
    for i in range(9):
      # mu
      z_use = z_mu_pca[labels == i]
      ax[0].plot(z_use[::skip_factor, 0], z_use[::skip_factor, 1],
                linestyle='', marker='.')
      z_mean = np.mean(z_use, 0)
      ax[2].plot([z_mean[0]], [z_mean[1]],
                  linestyle='', marker='.')
      # sigma
      z_use = z_sigma_pca[labels == i]
      ax[1].plot(z_use[::skip_factor, 0], z_use[::skip_factor, 1],
              linestyle='', marker='.')
      z_mean = np.mean(z_use, 0)
      ax[3].plot([z_mean[0]], [z_mean[1]],
                  linestyle='', marker='.')
    ax[2].set_xlim(ax[0].get_xlim())
    ax[2].set_ylim(ax[0].get_ylim())
    ax[3].set_xlim(ax[1].get_xlim())
    ax[3].set_ylim(ax[1].get_ylim())
    #
    ax[0].set_title('mu')
    ax[1].set_title('sigma')
    ax[2].set_title('E(mu)')
    ax[3].set_title('E(sigma)')
    fig2.canvas.draw()
    plt.show(block=False)
    print()

    # INTERPOLATE
    z_mu_mean = np.zeros((10, dim_z), dtype=np.float64)
    z_sigma_mean = np.zeros((10, dim_z), dtype=np.float64)
    for i in range(10):
      z_use = z_mu[labels == i]
      z_mu_mean[i] = np.mean(z_use, 0)
      z_use = z_sigma[labels == i]
      z_sigma_mean[i] = np.mean(z_use, 0)

    nInterp = 5
    z_mu_interp = np.zeros((10, nInterp, dim_z),
                            dtype=np.float64)
    # z_sigma_interp = np.zeros((10, nInterp, dim_z),
    #                           dtype=np.float64)
    for i in range(10):
      # mu
      if (i==9):
        z0 = z_mu_mean[-1]
        z1 = z_mu_mean[0]
      else:
        z0 = z_mu_mean[i]
        z1 = z_mu_mean[i+1]
      for j in range(dim_z):
        z_mu_interp[i, :, j] = np.linspace(z0[j], z1[j], nInterp+1)[:-1]
      # # sigma
      # if (i==9):
      #   z0 = z_sigma_mean[-1]
      #   z1 = z_sigma_mean[0]
      # else:
      #   z0 = z_sigma_mean[i]
      #   z1 = z_sigma_mean[i+1]
      # for j in range(dim_z):
      #   z_sigma_interp[i, :, j] = np.linspace(z0[j], z1[j], nInterp+1)[:-1]
    z_mu_interp = torch.from_numpy(z_mu_interp).to(dev)
    # z_sigma_interp = torch.from_numpy(z_sigma_interp).to(dev)

    # PLOT INTERPOLATION
    print('Plotting number interpolation:')
    fig3 = plt.figure(3, figsize=(8, 16))
    fig3.clear()
    ax = list()
    nRows = 10
    nCols = nInterp
    nPlots = nRows * nCols
    for i in range(nPlots):
      ax.append(fig3.add_subplot(nRows,nCols,i+1))
      ax[-1].clear()
      ax[-1].axis('off')
    for i in range(nRows):
      for j in range(nCols):
        z_use = (z_mu_interp[i, j])[None, :]
        x_mu_interp = vae.decode(z_use).detach().cpu().numpy()
        ax[i*nCols+j].imshow(x_mu_interp[0], cmap='gray', vmin=0.0, vmax=1.0)
    fig3.canvas.draw()
    plt.show(block=False)
    print()
    #
    print('Finished analyzing VAE')
    print()
