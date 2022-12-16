import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse
from load_model import load_resnet

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans_fixed.projected_gradient_descent import (
    projected_gradient_descent,
)

# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AdversarialDataset(Dataset):
    def __init__(self, args, root='./custom_data', train=True, mixed=True, add_gaussian=False):
        if not os.path.exists(root):
            raise ValueError("Root directory does not exist.")
        # mixing different adversarial datasets
        if mixed:
            if train:
                path0 = os.path.join(root, f"{args.dataset}_train.pt")
                path1 = os.path.join(root, f"{args.dataset}_gaussian_train.pt")
                path2 = os.path.join(root, f"{args.dataset}_fgsm_norm{args.norm}_train.pt")
                path3 = os.path.join(root, f"{args.dataset}_pgd_norm{args.norm}_train.pt")
                
            else:
                path0 = os.path.join(root, f"{args.dataset}_test.pt")
                path1 = os.path.join(root, f"{args.dataset}_gaussian_test.pt")
                path2 = os.path.join(root, f"{args.dataset}_fgsm_norm{args.norm}_test.pt")
                path3 = os.path.join(root, f"{args.dataset}_pgd_norm{args.norm}_test.pt")
            # load pre-generated dataset from path
            gt = torch.load(path0)
            gaussian_dataset = torch.load(path1)
            fgsm_datset = torch.load(path2)
            pgd_dataset = torch.load(path3)
            if add_gaussian: # add gaussian noise dataset
                self.clean = torch.cat([gt, gt, gt], dim=0)
                self.noisy = torch.cat([fgsm_datset, pgd_dataset, gaussian_dataset], dim=0)
                print("Loaded Datasets:\n{}\n{}\n{}".format(path1, path2, path3))
            else:
                self.clean = torch.cat([gt, gt], dim=0)
                self.noisy = torch.cat([fgsm_datset, pgd_dataset], dim=0)
        # using a single dataset
        else: 
            if train:
                gt = f"{args.dataset}_train.pt"
                adv = args.train_file_name
            else:
                gt = f"{args.dataset}_test.pt"
                adv = args.test_file_name

            self.clean = torch.load(os.path.join(root, gt))
            self.noisy = torch.load(os.path.join(root, adv))

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):        
        return self.clean[idx], self.noisy[idx]


def img_to_numpy(x):
    return np.clip(x.detach().cpu().numpy().squeeze().transpose(1, 2, 0), 0., 1.)

def get_dataset(dataset, sample_test=False):
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        train_data = MNIST(root='./data', train=True, download=True, transform=transform)
        test_data = MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)     
    else:
        raise ValueError("Dataset not supported.")

    if sample_test: #subsample 1000 images for testset
        # select 1000 random samples
        torch.manual_seed(0)
        indices = torch.randperm(len(test_data))[:1024]
        test_data = torch.utils.data.Subset(test_data, indices)
        
    return train_data, test_data

def get_dataloader(dataset, batch_size, sample_test=False):
    train_data, test_data = get_dataset(dataset, sample_test=sample_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def generate_attack(model, images, attack_mode, eps_range, grayscale=False):
    # generate random noise level within given range
    eps_random = np.random.uniform(eps_range[0], eps_range[-1])
    if attack_mode == 'gaussian':
        if grayscale: 
            noise = torch.randn_like(images[:, 0, :, :].unsqueeze(1)) * eps_random
        else: 
            noise = torch.randn_like(images) * eps_random
        images = torch.clamp(images + noise, min=0.0, max=1.0)
    elif attack_mode == 'fgsm':
        images = fast_gradient_method(
            model_fn=model,
            x=images,
            eps=eps_random,
            norm=args.norm,
            clip_min=0.0,
            clip_max=1.0,
        )
    elif attack_mode == 'pgd':
        images = projected_gradient_descent(
            model_fn=model,
            x=images,
            eps=eps_random,
            eps_iter=0.005,
            nb_iter=40,
            norm=args.norm,
            clip_min=0.0,
            clip_max=1.0,
            sanity_checks=False
        )
    elif attack_mode == 'cw':
        images = carlini_wagner_l2(
            model_fn=model,
            x=images,
            n_classes=10,
            max_iterations=10
        )
    else:
        raise ValueError("attack mode {} not supported".format(attack_mode))

    return images


def generate_adv_examples(args, model):

    train_loader, test_loader = get_dataloader(args.dataset, args.batch_size)
    train_list = []
    test_list = []

    if args.adv_mode == 'none':
        # clean set
        for (images, _) in train_loader: train_list.append(images)
        for (images, _) in test_loader: test_list.append(images)
    else:
        # convert to int
        if args.norm == 'inf':
            args.norm = np.inf
        elif args.norm == '1' or args.norm == '2':
            args.norm = int(args.norm)
        else:
            raise ValueError("Norm not supported")

        # generate attack examples    
        for (images, _) in tqdm(train_loader):
            images = images.to(device)
            images = generate_attack(model, images, args.adv_mode, args.eps_range, grayscale=(args.dataset == 'mnist'))
            train_list.append(images.detach().cpu())
        
        for (images, _) in tqdm(test_loader):
            images = images.to(device)
            images = generate_attack(model, images, args.adv_mode, args.eps_range, grayscale=(args.dataset == 'mnist'))
            test_list.append(images.detach().cpu())
        
    train_tensor = torch.cat(train_list, dim=0)
    test_tensor = torch.cat(test_list, dim=0)

    if not os.path.isdir('custom_data'): os.mkdir('custom_data')

    torch.save(train_tensor, os.path.join("custom_data", args.train_file_name))
    torch.save(test_tensor, os.path.join("custom_data", args.test_file_name))


def test_dataset(args, ds):
    idx = np.random.randint(0, 100)
    print(idx)
    clean = img_to_numpy(ds[idx][0])
    noisy = img_to_numpy(ds[idx][1])
    plt.figure()
    plt.subplot(211)
    plt.imshow(clean)
    plt.title(f"clean vs. {args.adv_mode} at index {idx}")
    plt.subplot(212)
    plt.imshow(noisy)
    plt.show()


if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train on. One of [mnist, cifar10].')
    parser.add_argument('--adv_mode', type=str, default='fgsm', help='type of adversarial noise')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--norm', type=str, default='inf', help='Norm to use for attack (if possible to set). One of [inf, 1, 2].')
    parser.add_argument('--peek', action='store_true', help='peek or generate')
    args = parser.parse_args()

    # epsilon range
    if args.norm == 'inf':
        if args.dataset == 'mnist':
            args.eps_range = [4 / 256, 16 / 256]
        elif args.dataset == 'cifar10':
            if args.adv_mode == 'gaussian':
                args.eps_range = [4 / 256, 14 / 256]
            else:
                args.eps_range = [4 / 256, 16 / 256]
    elif args.norm == '2':
        args.eps_range = [0.5, 3.5]

    # set file names to save/load
    if args.adv_mode == 'none':
        args.train_file_name = f"{args.dataset}_train.pt"
        args.test_file_name = f"{args.dataset}_test.pt"
    elif args.adv_mode == 'gaussian' or args.adv_mode == 'cw':
        args.train_file_name = f"{args.dataset}_{args.adv_mode}_train.pt"
        args.test_file_name = f"{args.dataset}_{args.adv_mode}_test.pt"
    else:
        args.train_file_name = f"{args.dataset}_{args.adv_mode}_norm{args.norm}_train.pt"
        args.test_file_name = f"{args.dataset}_{args.adv_mode}_norm{args.norm}_test.pt"

    # resnet classification model (used to generate adversarial examples)
    net = load_resnet(device=device, grayscale=(args.dataset == 'mnist'))
    net.load_state_dict(torch.load(os.path.join("trained_models", args.dataset, 'resnet18_2.0+0_BL.pth'), map_location=device))

    print("Generating Adversarial Examples ...")
    print(f"dataset: {args.dataset}")
    print(f"attack: {args.adv_mode}")
    print(f"norm: {args.norm}")
    print(f"eps range: {args.eps_range}")
    print("=======================================================")
    if not args.peek:
        # generate adversarial examples
        generate_adv_examples(args, net)
    else: 
        # verify the dataset
        ds = AdversarialDataset(args, train=False, mixed=(args.adv_mode == 'mixed'))
        test_dataset(args, ds)
