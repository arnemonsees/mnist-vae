# mnist-vae
Implementation of a variational autoencoder for generating MNIST samples

## Instructions
1. Open Google Colab: https://research.google.com/colaboratory/
2. Recommended: Request GPU (Runtime -> Change runtime type -> Hardware accelerator: GPU)
3. Upload: colab.ipynb, env.yml, mnist_io.py, vae.py
4. Run colab.ipynb cell by cell

## Results
The trained VAE can, for instance, be used to generate smooth interpolations between different numbers. This is achieved by moving along a trajectory in latent space, which connects the respective numbers.

![](https://github.com/arnemonsees/mnist-vae/blob/main/sample.png)
