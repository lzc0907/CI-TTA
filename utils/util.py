# coding=utf-8
import random
import numpy as np
import torch
import sys
import os
import torchvision
import PIL


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(filename, alg, args):
    save_dict = {
        "args": vars(args),
        "model_dict": alg.cpu().state_dict()
    }
    torch.save(save_dict, os.path.join(args.output, filename))
    alg.cuda()


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict


def alg_loss_dict(args):
    loss_dict = {'ANDMask': ['total'],
                 'CORAL': ['class', 'coral', 'total'],
                 'DANN': ['class', 'dis', 'total'],
                 'ERM': ['class'],
                 'Mixup': ['class'],
                 'MLDG': ['total'],
                 'MMD': ['class', 'mmd', 'total'],
                 'GroupDRO': ['group'],
                 'RSC': ['class'],
                 'VREx': ['loss', 'nll', 'penalty'],
                 'DIFEX': ['class', 'dist', 'exp', 'align', 'total']
                 }
    return loss_dict[args.algorithm]


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'office':
        domains = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-caltech':
        domains = ['amazon', 'dslr', 'webcam', 'caltech']
    elif dataset == 'office-home':
        domains = ['Art', 'Clipart', 'Product', 'RealWorld']
    elif dataset == 'dg5':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    elif dataset == 'PACS':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'PAC':
        domains = ['art_painting', 'cartoon', 'photo']
    elif dataset == 'PAC_mix_p':
        domains = ['art_painting', 'art_painting_c','cartoon','cartoon_c', 'photo']
    elif dataset == 'PAC_mix_c':
        domains = ['art_painting', 'art_painting_c',  'photo', 'photo_c','cartoon']
    # elif dataset == 'P_edge':
    #     domains = ['photo', 'photo_canny', 'sketch']
    elif dataset == 'P':
        domains = ['photo', 'cartoon']
    elif dataset == 'A':
        domains = ['art_painting', 'cartoon']
    elif dataset == 'C':    
        domains = ['cartoon', 'art_painting']
    elif dataset == 'S':
        domains = ['sketch', 'art_painting']    
    elif dataset == 'P_edge':
        domains = ['photo', 'sketch']
    elif dataset == 'pacs_c':
        domains = ['art_painting_c', 'cartoon_c', 'photo_c', 'sketch_c']
    elif dataset == 'PAC_mix_a':
        domains = ['cartoon', 'cartoon_c_r', 'photo', 'photo_c_r','art_painting']
    elif dataset == 'VLCS':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset == 'domainnet':
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'office': ['amazon', 'dslr', 'webcam'],
        'office-caltech': ['amazon', 'dslr', 'webcam', 'caltech'],
        'office-home': ['Art', 'Clipart', 'Product', 'RealWorld'],
        'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'],
        'PAC': ['art_painting', 'cartoon', 'photo'],
        'PAC_mix_p': ['art_painting', 'art_painting_c', 'cartoon', 'cartoon_c', 'photo'],
        'PAC_mix_c': ['art_painting', 'art_painting_c', 'photo', 'photo_c', 'cartoon'],
        'PAC_mix_a': ['cartoon', 'cartoon_c_r', 'photo', 'photo_c_r','art_painting'],
        'pacs_c': ['art_painting_c', 'cartoon_c', 'photo_c', 'sketch_c'],
        # 'P_edge': ['photo', 'photo_canny', 'sketch'],
        'P': ['photo', 'cartoon'],
        'A': ['art_painting', 'cartoon'],
        'C': ['cartoon', 'art_painting'],
        'S': ['sketch', 'art_painting'],
        'P_edge': ['photo', 'sketch'],
        'dg5': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
        'VLCS': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
        'domiannet': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    }
    if dataset == 'dg5':
        args.input_shape = (3, 32, 32)
        args.num_classes = 10
    else:
        args.input_shape = (3, 224, 224)
        if args.dataset == 'office-home':
            args.num_classes = 65
        elif args.dataset == 'office':
            args.num_classes = 31
        elif args.dataset == 'PACS' or args.dataset == 'PAC' or args.dataset == 'PAC_mix_p' or args.dataset == 'PAC_mix_c' or args.dataset == 'PAC_mix_a' or args.dataset == 'P_edge':
            args.num_classes = 7
        elif args.dataset == 'VLCS':
            args.num_classes = 5
        elif args.dataset == 'domiannet':
            args.num_classes = 345
        else:
            args.num_classes = 7
    return args
