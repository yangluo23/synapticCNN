import os
import torch
import torch.nn as nn
import tqdm
import argparse
from model import *
from dataset import prepare_mnist_dataloader, \
                    prepare_cifar10_dataloader, prepare_cifar100_dataloader
from model.factory import set_random_seeds, save_model
import pandas as pd

def evaluate_model(model, test_loader, device, criterion=None):

    print("evaluating ...")
    running_loss = 0
    running_corrects = 0

    for i, (inputs, labels) in enumerate(tqdm.tqdm(test_loader, desc='Evaluating')):

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

def evaluate_rnn_model(rnn, test_loader, device, criterion=None):

    print("evaluating ...")

    running_loss = 0
    running_corrects = 0

    for i, (inputs, labels) in enumerate(tqdm.tqdm(test_loader, desc='Evaluating')):

        inputs = inputs.to(device)
        labels = labels.to(device)

        hidden = rnn.initHidden().to(device)

        for index in range(inputs.shape[1]):
            outputs, hidden = rnn(inputs[:, index, :, :], hidden)

        _, preds = torch.max(outputs.squeeze(0), 1)

        if criterion is not None:
            loss = criterion(outputs.squeeze(0), labels)
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy


if __name__ == '__main__':

    parse = argparse.ArgumentParser(description='test synaptic CNN')

    parse.add_argument('-d', '--data', choices=["cifar", "mnist", "cifar100"], type=str, default='mnist',
                       help="choice your dataset, only support 'cifar', 'cifar100 or 'mnist', default: mnist")
    parse.add_argument('-p', '--save-path', default="./test_results", type=str,
                       help="filepath where to save your test results")
    parse.add_argument('-w', '--workers', default=8, type=int,
                       help="num workers")
    parse.add_argument('-b', '--bsz', default=64, type=int,
                       help="test batch size")
    parse.add_argument('-l','--load-path', type=str, default='',
                       help="your saved pth file path for model loading")
    parse.add_argument('-c', '--criterion', default=False, type=bool,
                       help="test loss function")
    parse.add_argument('--ckpt', required=True, type=str,
                       help="checkpoint file path '*.pkl' or '*.pth'")
    parse.add_argument('--codebook', required=True, type=str,
                       help="codebook file path '*.xlsx'")
    parse.add_argument('--all', default=False, type=bool,
                       help="all layers together quantization")
    parse.add_argument('--multi-devices', default=False, type=bool,
                       help="whether to use multi devices mapping")
    parse.add_argument('--num', default=1, type=int,
                       help="device/codebook's number")
    parse.add_argument('--stopping-temperature', default=1e-8, type=float,
                       help="stopping temperature")
    parse.add_argument('--search-range', default=0.001, type=float,
                       help="search range")
    parse.add_argument('--waiting-steps', default=100, type=int,
                       help="iteration per temp")
    parse.add_argument('--quantize', default=True, type=bool,
                       help="whether do quantization")
    parse.add_argument('--save', default=True, type=bool,
                       help="whether to save quantized model")
    parse.add_argument('--save-dir', type=str, default='./Qckpt_v2',
                       help="checkpoint dir path")
    parse.add_argument('--filename', type=str, default='',
                       help="checkpoint file name")
    parse.add_argument('--model', type=str, default='',
                       choices=['vgg6', 'vgg8', 'vgg11', 'vgg13','vgg16',
                        'cnn3', 'cnn5', 'cnn7',
                        'res18', 'res34', 'res50tiny'], help="model name")

    parse.add_argument('-s', '--seed', default=0, type=int,
                       help="random seed")
    parse.add_argument('--comments', type=str,
                       default='',
                       help="experiment's comments")

    args = parse.parse_args()
    set_random_seeds(args.seed)

    # data
    test_loader = None
    if args.data == 'mnist':
        _, test_loader = prepare_mnist_dataloader(num_workers=args.workers,
                                                  eval_batch_size=args.bsz)
    elif args.data == 'cifar':
        _, test_loader = prepare_cifar10_dataloader(num_workers=args.workers,
                                                  eval_batch_size=args.bsz)
    elif args.data == 'cifar100':
        _, test_loader = prepare_cifar100_dataloader(num_workers=args.workers,
                                                    eval_batch_size=args.bsz)
    else:
        raise KeyError("invalid dataset type")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # codebooks
    device_number = args.num
    df = pd.read_excel(args.codebook).values
    codebook = torch.from_numpy(df[:, device_number]).unsqueeze(1).to(device, dtype=torch.float)

    codebooks = None
    if args.multi_devices:
        codebooks = torch.from_numpy(df).to(device, dtype=torch.float)

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    show_scale_path = os.path.join('./show_V2', args.data, 'device_scale/randn_init', args.model + '_' + args.comments)

    # model
    if args.data == 'mnist':
        if args.model == 'cnn3':
            model = QBasicCNN(
                codebook=codebook,
                quantize=args.quantize,
                all_layers=args.all,
                device_number=device_number,
                devices=codebooks,
                show_scale=show_scale_path)
        elif args.model == 'cnn5':
            model = QBasicCNN5(
                codebook=codebook,
                quantize=args.quantize,
                all_layers=args.all,
                device_number=device_number,
                devices=codebooks,
                show_scale=show_scale_path)
        elif args.model == 'cnn7':
            model = QBasicCNN7(
                codebook=codebook,
                quantize=args.quantize,
                all_layers=args.all,
                device_number=device_number,
                devices=codebooks,
                show_scale=show_scale_path)
        else:
            raise KeyError("model name error.")
    # cifar10
    elif args.data == 'cifar':
        if args.model == 'vgg6':
            model = Qvgg6(
                codebook=codebook,
                quantize=args.quantize,
                all_layers=args.all,
                device_number=device_number,
                show_scale=show_scale_path)
        elif args.model == 'vgg8':
            model = Qvgg8(
                codebook=codebook,
                quantize=args.quantize,
                all_layers=args.all,
                device_number=device_number,
                show_scale=show_scale_path)
        elif args.model == 'vgg11':
            model = Qvgg11(
                codebook=codebook,
                quantize=args.quantize,
                all_layers=args.all,
                device_number=device_number,
                show_scale=show_scale_path)
        elif args.model == 'vgg13':
            model = Qvgg13(
                codebook=codebook,
                quantize=args.quantize,
                all_layers=args.all,
                device_number=device_number,
                show_scale=show_scale_path)
        elif args.model == 'vgg16':
            model = Qvgg16(
                codebook=codebook,
                quantize=args.quantize,
                all_layers=args.all,
                device_number=device_number,
                show_scale=show_scale_path)
        else:
            raise KeyError("model name error.")

    elif args.data == 'cifar100':
        if args.model == 'res18':
            model = Qresnet18(
                codebook=codebook,
                quantize=args.quantize,
                all_layers=args.all,
                device_number=device_number,
                stopping_temperature=args.stopping_temperature,
                search_range=args.search_range,
                waiting_steps=args.waiting_steps,
                num_classes=100,
                show_scale=show_scale_path)
        elif args.model == 'res34':
            model = Qresnet34(
                codebook=codebook,
                quantize=args.quantize,
                all_layers=args.all,
                device_number=device_number,
                stopping_temperature=args.stopping_temperature,
                search_range=args.search_range,
                waiting_steps=args.waiting_steps,
                num_classes=100,
                show_scale=show_scale_path)
        elif args.model == 'res50tiny':
            model = Qresnet50tiny(
                codebook=codebook,
                quantize=args.quantize,
                all_layers=args.all,
                device_number=device_number,
                stopping_temperature=args.stopping_temperature,
                search_range=args.search_range,
                waiting_steps=args.waiting_steps,
                num_classes=100,
                show_scale=show_scale_path)

    else:
        raise KeyError("model name error.")

    print(model)
    model = load_model(model, args.ckpt, device)
    model.to(device)

    # quantize
    model.eval()

    if args.save:
        save_dir = os.path.join(args.save_dir+args.comments, args.data, args.model)

        save_model(model, model_dir=save_dir,
                   model_filename="Device-{}_{}-{}-all_layer-{}.pkl".format(device_number,
                                                                            args.filename,
                                                                            os.path.basename(args.ckpt).split('.')[0],
                                                                  'True' if args.all else 'False'))

    # criterion
    criterion = None
    if args.criterion:
        criterion = nn.CrossEntropyLoss()

    # evaluate
    loss, acc = evaluate_model(model,test_loader, device, criterion)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        loss, 100. * acc))
