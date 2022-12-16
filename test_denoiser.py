import torch
import numpy as np
from tqdm import tqdm
import os
import argparse

from models.denoisers import DnCNN, ConvDAE
from adversarial_dataset import get_dataloader
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans_fixed.projected_gradient_descent import (
    projected_gradient_descent,
)
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from train_denoiser import img_to_numpy
from load_model import load_resnet

# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_attack(images, model, attack, norm, eps):
    # perturb the images with various attack methods
    if attack == 'fgsm':
        x_adv = fast_gradient_method(
            model_fn=model,
            x=images,
            eps=eps,
            norm=norm,
            clip_min=0.0,
            clip_max=1.0,
        )
    elif attack == 'pgd':
        x_adv = projected_gradient_descent(
            model_fn=model,
            x=images,
            eps=eps,
            eps_iter=5e-4,
            nb_iter=40,
            norm=norm,
            clip_min=0.0,
            clip_max=1.0,
            sanity_checks=False
        )
    elif attack == 'cw':
        x_adv = carlini_wagner_l2(
            model_fn=model,
            x=images,
            n_classes=10,
            lr=5e-3,
            binary_search_steps=5,
            max_iterations=100,
            initial_const=1e-3
        )
        # keep track of the l2 norm of the perturbed images
        l2_norm = torch.norm((images - x_adv).view(images.shape[0], -1), p=2, dim=1)
        indices = (l2_norm > eps)
        # ignore images with l2 norm larger than eps
        x_adv[indices] = images[indices]

    else: x_adv = images

    return x_adv


def evaluate_metrics(model, denoisers, test_loader, attack, norm, eps):
    #denoiser.eval()
    for denoiser in denoisers: denoiser.eval()
    n = len(denoisers)
    avg_psnr_bl=0.0
    avg_ssim_bl=0.0
    total=0
    # avg_psnr=0
    # avg_ssim=0

    # initialize dict
    denoiser_metrics = [{'PSNR': 0.0, 'SSIM': 0.0} for _ in range(n)]
    
    for images, _ in tqdm(test_loader):
        images = images.to(device)
        x_adv = generate_attack(images, model, attack, norm, eps)

        with torch.no_grad():
            #output = denoiser(x_adv)
            denoised = []
            for i, denoiser in enumerate(denoisers):
                output = denoiser(x_adv)
                denoised.append( output )

            for i in range(len(images)):
                original = img_to_numpy(images[i])
                adv = img_to_numpy(x_adv[i])
                # PSNR and SSIM of adversarial images
                avg_psnr_bl += PSNR(original, adv)
                avg_ssim_bl += SSIM(original, adv, channel_axis=-1)
                # PSNR and SSIM of denoised images
                # denoised = img_to_numpy(output[i])
                # avg_psnr += PSNR(original, denoised)
                # avg_ssim += SSIM(original, denoised, channel_axis=-1)
                for j in range(n): 
                    denoised_j = img_to_numpy(denoised[j][i])
                    denoiser_metrics[j]['PSNR'] += PSNR(original, denoised_j)
                    denoiser_metrics[j]['SSIM'] += SSIM(original, denoised_j, channel_axis=-1)

            total += len(images)

    avg_psnr_bl /= total      
    avg_ssim_bl /= total  
    # avg_psnr /= total
    # avg_ssim /= total
    print("Baseline: \nAverage PSNR:{:.3f} \nAverage SSIM: {:.3f}".format(avg_psnr_bl, avg_ssim_bl))
    # print("\nAverage PSNR:{:.3f} \nAverage SSIM: {:.3f}".format(avg_psnr, avg_ssim))

    for j in range(n): 
        denoiser_metrics[j]['PSNR'] /= total
        denoiser_metrics[j]['SSIM'] /= total
        print("Denoiser{}: (Average PSNR:{:.3f} , Average SSIM: {:.3f})".format(j + 1, denoiser_metrics[j]['PSNR'], denoiser_metrics[j]['SSIM']))
    
    # return avg_psnr_bl, avg_ssim_bl, avg_psnr, avg_ssim, total
    return total, avg_psnr_bl, avg_ssim_bl, denoiser_metrics


def test_acc(model, denoisers, test_loader, attack, norm, eps):
    #resnet model
    model.eval()
    #denoisers to test
    for denoiser in denoisers: denoiser.eval()

    n = 2 + len(denoisers) 
    acc_list = np.zeros((n))
    total = 0.0
    
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        # generate attack
        x_adv = generate_attack(images, model, attack, norm, eps)
        
        total += labels.size(0)

        with torch.no_grad():
            # baseline
            outputs = model(images)
            # attack
            adv_outputs = model(x_adv)

            _, pred = torch.max(outputs, 1)
            _, pred_adv = torch.max(adv_outputs, 1)

            # baseline and adversarial accuracy
            acc_list[0] += (pred == labels).sum().item()
            acc_list[1] += (pred_adv == labels).sum().item()

            #denoisers
            for i, denoiser in enumerate(denoisers):
                denoised = denoiser(x_adv)
                denoised_outputs = model(denoised)
                _, pred_dn = torch.max(denoised_outputs, 1)
                acc_list[i+2] += (pred_dn == labels).sum().item()

    # compute the accuracy over all test images
    acc_list = (acc_list / total)
    print("Test Accuracy no attack: {}".format(acc_list[0]))
    print("Test Accuracy with {} attack: {}".format(attack, acc_list[1]))
    for i in range(len(denoisers)):
        print("Test Accuracy with {} attack + denoiser{}: {}".format(attack, i+1, acc_list[i+2]))

    return acc_list


if __name__ == '__main__':

    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train on. One of [mnist, cifar10].')
    parser.add_argument('--arch', type=str, default='dncnn', help='Dataset to train on. One of [dncnn, dae1, dae2].')
    parser.add_argument('--denoiser', type=str, default='mixed+gaussian', help='')
    parser.add_argument('--adv_mode', type=str, default='fgsm', help='type of adversarial noise')
    parser.add_argument('--eps', type=float, default=16/256, help='perturbation level')
    parser.add_argument('--norm', type=str, default='inf', help='Norm to use for attack (if possible to set). One of [inf, 1, 2].')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    args = parser.parse_args()

    # load data for testing
    _, test_loader = get_dataloader(args.dataset, args.batch_size, sample_test=False)

    # denoiser model
    if args.arch == 'dncnn':
        denoiser = DnCNN(in_channels=3, out_channels=3, depth=7, hidden_channels=64, use_bias=False).to(device)
    elif args.arch == 'dae':
        denoiser = ConvDAE(in_channels=3, out_channels=3, use_bias=False).to(device)

    # load pretrained denoiser
    denoiser_name = f"{args.arch}_{args.dataset}_{args.denoiser}.pth"
    denoiser_path = './trained_denoisers/' + denoiser_name
    denoiser.load_state_dict(torch.load(denoiser_path, map_location=device))

    #load classification model
    net = load_resnet(device=device, grayscale=(args.dataset == 'mnist'))
    net.load_state_dict(torch.load(os.path.join("trained_models", args.dataset, 'resnet18_2.0+0_BL.pth'), map_location=device))

    # convert norm
    if args.norm == 'inf':
        args.norm = np.inf
    elif args.norm == '1' or args.norm == '2':
        args.norm = int(args.norm)
    else:
        raise ValueError("Norm not supported")
    
    print("=================== Testing ====================")
    print(f"denoiser: {denoiser_name}")
    print(f"dataset: {args.dataset}")
    print(f"noise type: {args.adv_mode}")
    print(f"eps: {args.eps}")
    print(f"norm: {args.norm}")
    print("=======================================================")
    denoisers = [denoiser]
    # test accuracy of model
    test_acc(net, denoisers, test_loader, attack=args.adv_mode, eps=args.eps, norm=args.norm)
