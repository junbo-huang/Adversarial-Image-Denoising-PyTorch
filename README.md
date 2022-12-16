# Adversarial-Image-Denoising-PyTorch

Denoiser training and testing code for CSC2529 Project. Main project repository: https://github.com/JasonTang99/csc2529_project

### Denoiser
- ```trained_denoisers/```: holds pretrained denoiser models.
- ```models/denoisers.py```: defines denoiser architectures (DnCNN, Convolutional Autoencoder).
- ```adversarial_dataset.py```: loads and generates custom MNIST and CIFAR10 adversarial dataset for training.
- ```test_denoiser.py```: test denoiser performance by evaluating model accuracy, PSNR, and SSIM of reconstructed images. Also generate FGSM, PGD and CW attacks.
- ```plot_denoiser_exp.ipynb```: plot visualization and results on denoisers
- ```train_denoiser.py```: trains the denoiser models with user-specified parameters. 

Sample Usage:
```
python train_denoiser.py --dataset=cifar10 --arch=dncnn --lr=1e-3 --batch_size=64 --epochs=5
python test_denoiser.py
python adversarial_dataset.py
```
