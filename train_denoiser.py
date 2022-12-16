import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse

from models.denoisers import DnCNN, ConvDAE
from adversarial_dataset import AdversarialDataset, img_to_numpy
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def add_gaussian_noise(x, noise_level):
    noisy = x + torch.randn_like(x) * noise_level
    return noisy

def calc_psnr(x, gt):
    out = 10 * np.log10(1 / ((x - gt)**2).mean().item())
    return out

def show_batch(images, noisy, denoised, n=6):
    plt.figure(figsize=(20, 10), dpi=500)

    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i+1)
        img = img_to_numpy(images[i])
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noisy image
        ax = plt.subplot(3, n, i +1 + n)
        img = img_to_numpy(noisy[i])
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display denoised image
        ax = plt.subplot(3, n, i +1 + n + n)
        img = img_to_numpy(denoised[i])
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.figtext(0.5,0.95, "ORIGINAL IMAGES", ha="center", va="top", fontsize=14, color="b")
    plt.figtext(0.5,0.65, "NOISY IMAGES", ha="center", va="top", fontsize=14, color="b")
    plt.figtext(0.5,0.35, "DENOISED RECONSTRUCTED IMAGES", ha="center", va="top", fontsize=14, color="b")
    plt.subplots_adjust(hspace = 0.5 )
    plt.show()


def evaluate_model(model, data_loader, test=False, show=False):
    if test: model.eval()

    avg_psnr=0
    avg_ssim=0
    total=0
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            noisy = labels
            output = model(noisy)
            for i in range(len(images)):
                original = img_to_numpy(images[i])
                denoised = img_to_numpy(output[i])
                avg_psnr += PSNR(original, denoised)
                avg_ssim += SSIM(original, denoised, channel_axis=-1)

            total += len(images)
    
    if show: show_batch(images, noisy, output, n=10)

    print("\nAverage PSNR:{:.3f} \nAverage SSIM: {:.3f}".format(avg_psnr/total, avg_ssim/total))


def train(args, model, train_loader, val_loader, use_scheduler=False):

    # MSE loss function 
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if use_scheduler:
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        lambda1 = lambda epoch: 0.7 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    losses = []
    model.train()
    pbar = tqdm(total=len(train_loader) * args.epochs)
    for epoch in range(1, args.epochs + 1):

        train_loss = 0.0
        for idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            noisy_imgs = labels      
            optimizer.zero_grad()
            # denoiser
            denoised = model(noisy_imgs)

            loss = criterion(denoised, images)
            loss.backward()
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * images.size(0)
            losses.append(loss.item())

            if not idx % 200:
                psnr = calc_psnr(denoised, images)
                baseline_psnr = calc_psnr(noisy_imgs, images)
                print("\nTraining Loss: {:.4f} | Baseline PSNR: {:.2f} | PSNR: {:.2f}".format(train_loss/(idx + 1), baseline_psnr, psnr))

            pbar.update(1)
        
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('\nEpoch: {} | Training Loss: {:.4f}'.format(epoch, train_loss))
        # evaluate model on validation set
        # evaluate_model(model, val_loader, args.adv_mode, test=False) 
        if use_scheduler: 
            scheduler.step()
            print("Learning Rate: ", optimizer.param_groups[0]['lr'])

    #show_batch(images, noisy_imgs, denoised, n=10)
    pbar.close()
    return losses


if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='dncnn', help='Dataset to train on. One of [dncnn, dae].')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train on. One of [mnist, cifar10].')
    parser.add_argument('--adv_mode', type=str, default='gaussian', help='type of adversarial noise')
    parser.add_argument('--norm', type=str, default='inf', help='Norm to use for attack (if possible to set). One of [inf, 1, 2].')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--add_gaussian', action='store_true', help='')
    parser.add_argument('--use_scheduler', action='store_true', help='scheduler')
    parser.add_argument('--use_bias', action='store_true', help='')
    parser.add_argument('--use_bn', action='store_true', help='')

    args = parser.parse_args()

    if args.adv_mode == 'gaussian' or args.adv_mode == 'cw':
        args.train_file_name = f"{args.dataset}_{args.adv_mode}_train.pt"
        args.test_file_name = f"{args.dataset}_{args.adv_mode}_test.pt"
    else:
        args.train_file_name = f"{args.dataset}_{args.adv_mode}_norm{args.norm}_train.pt"
        args.test_file_name = f"{args.dataset}_{args.adv_mode}_norm{args.norm}_test.pt"

    # load adversarial dataset
    train_data = AdversarialDataset(args, train=True, mixed=(args.adv_mode == 'mixed'), add_gaussian=(args.add_gaussian))
    test_data = AdversarialDataset(args, train=False, mixed=(args.adv_mode == 'mixed'), add_gaussian=(args.add_gaussian))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    print("Train Size: ", len(train_loader.dataset))
    print("Test Size: ", len(test_loader.dataset))

    # denoiser model to train
    if args.arch == 'dncnn':
        model = DnCNN(in_channels=3, out_channels=3, depth=7, hidden_channels=64, use_bias=(args.use_bias)).to(device)
    elif args.arch == 'dae':
        model = ConvDAE(in_channels=3, out_channels=3, num_features=64, use_bias=(args.use_bias)).to(device)
   
    # model name to save
    model_name = f"{args.arch}_{args.dataset}_{args.adv_mode}{'+gaussian' if args.add_gaussian else ''}{'_bias' if args.use_bias else ''}.pth"
    

    # check if model exists
    if os.path.exists(os.path.join("trained_denoisers", model_name)):
        print("Model already exists. Skip Training ...")
        model.load_state_dict(torch.load(os.path.join("trained_denoisers", model_name), map_location=device))
    else:
        print("=================== Start Training ====================")
        print(f"model name: {model_name}")
        print(f"denoiser: {args.arch}")
        print(f"dataset: {args.dataset}")
        print(f"noise type: {args.adv_mode}")
        print(f"learning rate: {args.lr}")
        print(f"batch size: {args.batch_size}")
        print("=======================================================")
        # start training
        training_loss = train(args, model, train_loader, test_loader, use_scheduler=(args.use_scheduler))
        # create path if not exist
        if not os.path.isdir('trained_denoisers'): os.mkdir('trained_denoisers')
        # save model
        torch.save(model.state_dict(), os.path.join("trained_denoisers", model_name))

    # evaluate performance
    evaluate_model(model, test_loader, test=True, show=False)
