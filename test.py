import tqdm
import matplotlib.pyplot as plt
import itertools
# from sklearn.metrics import confusion_matrix

import torch.nn as nn
import argparse
from model import *
from dataset import prepare_mnist_dataloader, \
                    prepare_cifar10_dataloader, \
                    prepare_cifar100_dataloader

def plot_confusion_matrix(cm, classes,model_name,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.2)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{model_name}_after_mapping_confusion matrix.png', bbox_inches='tight')
    plt.show()

    return cm

def evaluate_model(model, test_loader, device, criterion=None, model_name=''):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0
    all_preds = torch.tensor([], device=device)

    for i, (inputs, labels) in enumerate(tqdm.tqdm(test_loader, desc="Evaluating")):

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds = torch.cat((all_preds, preds), dim=0)
        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    # cm = confusion_matrix(test_loader.dataset.targets, all_preds.data.cpu().numpy())
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # plt.figure(figsize=(10, 10))
    # cm = plot_confusion_matrix(cm, classes, model_name)
    # np.savetxt(f'{model_name}_after_mapping_confusion_matrix.csv', cm, delimiter=',')


    return eval_loss, eval_accuracy

def evaluate_rnn_model(rnn, test_loader, device, criterion=None):

    rnn.eval()
    rnn.to(device)

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

    parse.add_argument('-d', '--data', choices=["cifar", "mnist", "name", "cifar100"], type=str, default='mnist',
                       help="choice your dataset, only support 'cifar', 'cifar100', or 'mnist', default for mnist")
    parse.add_argument('-p', '--save-path', default="./test_results", type=str,
                       help="filepath where to save your test results")
    parse.add_argument('-w', '--workers', default=8, type=int,
                       help="num workers")
    parse.add_argument('-b', '--bsz', default=256, type=int,
                       help="test batch size")
    parse.add_argument('-l','--load-path', type=str, default='',
                       help="your saved pth file path for model loading")
    parse.add_argument('-c', '--criterion', default=False, type=bool,
                       help="test loss function")
    parse.add_argument('--ckpt', required=True, type=str,
                       help="checkpoint file path '*.pkl' or '*.pth'")
    parse.add_argument('-i', '--init', choices=["kaiming", "xavier", ""], default="",
                       help="different model's weight init methods, "
                            "only support 'kaiming', 'xavier' or 'uniform', default "
                            "method is uniform")
    parse.add_argument('--model', type=str, default='',
                       choices=['vgg6', 'vgg8', 'vgg11', 'vgg13', 'vgg16',
                                'cnn3', 'cnn5', 'cnn7',
                                'res18', 'res34', 'res50'], help="model name")

    args = parse.parse_args()


    # data & model
    test_loader = None
    if args.data == 'mnist':
        _, test_loader = prepare_mnist_dataloader(num_workers=args.workers,
                                                  eval_batch_size=args.bsz)
        if args.model == 'cnn3':
            model = BasicCNN(args.init)
        elif args.model == 'cnn5':
            model = BasicCNN5(args.init)
        elif args.model == 'cnn7':
            model = BasicCNN7(args.init)
        else:
            raise KeyError("invalid model name.")

    elif args.data == 'cifar':
        _, test_loader = prepare_cifar10_dataloader(num_workers=args.workers,
                                                    eval_batch_size=args.bsz)
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
        _, test_loader = prepare_cifar100_dataloader(num_workers=args.workers,
                                                    eval_batch_size=args.bsz)
        if args.model == 'res18':
            model = resnet18(pretrained=False, num_classes=100)
        elif args.model == 'res34':
            model = resnet34(pretrained=False, num_classes=100)
        elif args.model == 'res50':
            model = resnet50(pretrained=False, num_classes=100)

    else:
        raise KeyError("only support mnist, cifar10 and cifar100")

    print(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model, args.ckpt, device)

    # criterion
    criterion = None
    if args.criterion:
        criterion = nn.CrossEntropyLoss()

    # evaluate
    loss, acc = evaluate_model(model,test_loader, device, criterion)

    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {:.2f}%\n'.format(loss, 100. * acc))
