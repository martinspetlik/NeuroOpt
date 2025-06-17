# Neural Networks in Optics

 Repository for applications of neural networks in optics.

---

##  MNIST Diffusion Experiments

The `mnist_` directories contain scripts for training a **denoising diffusion model** on the MNIST dataset (handwritten digits). These scripts are **illustrative**, with a focus on the overall **structure and workflow** rather than achieving optimal performance.

### Running the Code

#### Simple Model

```bash
python model/mnist_diffusion/train_model_cnn.py \
  configs/mnist_diffusion/config_unet_simple.yaml \
  data_directory results_directory -c
```

#### Simple Model

```bash

python model/mnist_diffusion/train_model_cnn.py \
  configs/mnist_diffusion/config_unet_advanced.yaml \
  data_directory results_directory -c
```



----------------------------------
Some experiments exploring different activation functions for a fully connected neural network—designed to predict the mean and standard deviation of 10 variables sampled from a standard normal distribution \( N(0, 1) \) - can be found in `experiments/fully_connected_mean_std.py`.

