import os.path
import tqdm
import torch.nn as nn
import argparse
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import prepare_mnist_dataloader, \
                    prepare_cifar10_dataloader, prepare_cifar100_dataloader
from test import evaluate_model
from tools import *


def get_args():
    parse = argparse.ArgumentParser(description='pre-train synaptic CNN with full precision')

    parse.add_argument('-d', '--data', choices=["cifar", "mnist", "cifar100"], type=str, default='mnist',
                       help="choice your dataset, only support 'cifar', 'cifar100' or 'mnist', "
                            "default: mnist")
    parse.add_argument('-o', '--optimizer', choices=['adam', 'sgdm', ''], type=str,
                       default='adam',
                       help="choice your optimizer, only support 'adam' or 'sgdm', "
                            "default: adam")
    parse.add_argument('-p', '--save-path', default="./ckpt", type=str,
                       help="filepath where to save your trained models")
    parse.add_argument('--log', default="./logs", type=str,
                       help="log dir path where to record your training results")
    parse.add_argument('-e', '--epoch', default=10, type=int, help="training epochs")
    parse.add_argument('-w', '--workers', default=8, type=int,
                       help="num workers")
    parse.add_argument('-b', '--bsz', default=64, type=int,
                       help="training batch size")
    parse.add_argument('-t', '--eval-bsz', default=64, type=int,
                       help="evaluate batch size")
    parse.add_argument('-l', '--lr', default=1e-3, type=float,
                       help="learning rate")
    parse.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parse.add_argument('-sc', '--schedule', default=False, type=bool,
                       help="linear scheduler")
    parse.add_argument('--sc_type', default=2, type=int,
                       help="linear scheduler")
    parse.add_argument('-se', '--schedule-epoch', default=20, type=float,
                       help="linear scheduler epoch")
    parse.add_argument('-r', '--schedule-rescale', default=0.1, type=float,
                       help="linear scheduler weight")
    parse.add_argument('-i', '--init', choices=["kaiming", "xavier", ""], default="",
                       help="different model's weight init methods, "
                            "only support 'kaiming', 'xavier' or 'uniform', default: uniform")
    parse.add_argument('--model', type=str, default='',
                       choices=['vgg6', 'vgg8', 'vgg11', 'vgg13', 'vgg16',
                                'cnn3', 'cnn5', 'cnn7',
                                'res18', 'res34','res50'],
                       help="model name")
    parse.add_argument('-s', '--seed', default=1, type=int,
                       help="random seed")
    parse.add_argument( '--comments', default='', type=str,
                       help="expriments's comments")

    args = parse.parse_args()

    return args



def train_model():

    args = get_args()
    set_random_seeds(args.seed)

    # save pth
    save_dir = os.path.join(args.save_path, args.data)
    model_filename = "{}-epoch{}-{}-{}-{}.pkl".format(args.model, args.epoch,
                                                args.optimizer,
                                                "uniform" if not args.init else args.init, args.comments)
    writer_dir = os.path.join(args.log, args.data, ''.join(model_filename.split('.')[:-1]))
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    writer = SummaryWriter(writer_dir)


    # data & model
    if args.data == 'mnist':
        train_loader, test_loader = prepare_mnist_dataloader(num_workers=args.workers,
                                                  train_batch_size=args.bsz,
                                                  eval_batch_size=args.eval_bsz)
        if args.model == 'cnn3':
            model = BasicCNN(args.init)
        elif args.model == 'cnn5':
            model = BasicCNN5(args.init)
        elif args.model == 'cnn7':
            model = BasicCNN7(args.init)
        else:
            raise KeyError("invalid model name.")
    elif args.data == 'cifar':
        train_loader, test_loader = prepare_cifar10_dataloader(num_workers=args.workers,
                                                             train_batch_size=args.bsz,
                                                             eval_batch_size=args.eval_bsz)

        if args.model == 'vgg6':
            model = vgg6()
        elif args.model == 'vgg8':
            model = vgg8()
        elif args.model == 'vgg11':
            model = vgg11()
        elif args.model == 'vgg13':
            model = vgg13()
        elif args.model == 'vgg16':
            model = vgg16()
        else:
            raise KeyError("invalid model name.")
    elif args.data == 'cifar100':
        train_loader, test_loader = prepare_cifar100_dataloader(num_workers=args.workers,
                                                               train_batch_size=args.bsz,
                                                               eval_batch_size=args.eval_bsz)
        if args.model == 'res18':

            model = resnet18(pretrained=False, num_classes=100)

        elif args.model == 'res34':

            model = resnet34(pretrained=False, num_classes=100)

        elif args.model == 'res50':

            model = resnet50(pretrained=False, num_classes=100)

        else:
            raise KeyError("invalid model name.")
    else:
        raise KeyError("only support mnist, cifar10 and cifar100")

    print(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimizer
    weight_decay = 5e-4
    # if 'res' in args.model and args.data == 'cifar':
    #     weight_decay = 1e-4

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' \
            else torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)

    # linear scheduler
    milestones = [60, 120, 160]
    if args.sc_type == 1:
        milestones = [60, 160]
    # if 'res' in args.model and args.data == 'cifar':
    #     milestones = [100, 150]
    print(milestones)

    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    iter_per_epoch = len(train_loader)
    if args.data == "cifar100":
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # loss func
    criterion = nn.CrossEntropyLoss()

    # train
    best_eval_acc = 0.
    iteration = 0
    for epoch in range(args.epoch):

        model.train(True)

        if args.schedule and (epoch > 0 and epoch > args.warm):
            if args.data == "cifar100":
                train_scheduler.step(epoch-1)

        # if args.schedule and ((epoch + 1) % args.schedule_epoch) == 0:
        #     optimizer.param_groups[0]["lr"] *= args.schedule_rescale

        for step, (inputs, labels) in enumerate(tqdm.tqdm(train_loader, desc='Training')):

            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(dtype=torch.long, device=device)

            output = model(inputs)
            train_loss = criterion(output, labels)

            train_loss.backward()
            optimizer.step()

            if args.warm and (epoch <= args.warm) and args.data == "cifar100":
                warmup_scheduler.step()

            writer.add_scalar('lr', optimizer.param_groups[0]["lr"], iteration)
            iteration += 1

        if (args.epoch == epoch + 1) or (epoch % 2 == 0):
            train_acc = torch.sum(torch.max(output, 1)[1] == labels.data)
            eval_loss, eval_acc = evaluate_model(model, test_loader, device, criterion)
            print("Epoch: {:03d} Train Loss: {:.3f} Train Acc: "
                  "{:.3f} Eval Loss: {:.3f} "
                  "Eval Acc: {:.3f}".format(epoch,
                                            train_loss,
                                            train_acc,
                                            eval_loss,
                                            eval_acc))
            writer.add_scalar('loss/eval loss', eval_loss, iteration)
            writer.add_scalar('acc/eval acc', eval_acc * 100, iteration)
            writer.add_scalar('acc/train acc', train_acc, iteration)


            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                save_model(model, model_dir=save_dir, model_filename='best-' + model_filename)

        writer.add_scalar('loss/train loss', train_loss, iteration)

    # save model
    save_model(model, model_dir=save_dir, model_filename=model_filename)

    # plot
    parameters_analysis(model, args, filename=model_filename)

if __name__ == '__main__':
    train_model()

