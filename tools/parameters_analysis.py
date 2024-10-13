"""
This script is used to analyze the parameters of the quantized/mapped models.
"""
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import glob
import pandas as pd
import re
import statistics

from model import *

plt.rcParams["figure.figsize"] = [10, 10]
plt.rcParams["figure.autolayout"] = True

def get_args():
    parse = argparse.ArgumentParser(description='test synaptic CNN')

    parse.add_argument('-d', '--data', choices=["cifar", "mnist", "cifar100"], type=str, default='mnist',
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
    parse.add_argument('--quantize_results', type=str, required=False,
                       help="quantized weight results")
    parse.add_argument('--model', type=str, default='',
                       choices=['vgg6', 'vgg8', 'vgg11', 'vgg13', 'vgg16',
                                'cnn3', 'cnn5', 'cnn7',
                                'res18', 'res34', 'res50tiny'],
                       help="model name")

    args = parse.parse_args()

    return args

def parameters_analysis(model, args, save_fig='./show', show=True, filename='', result=''):

    weight = dict()
    bias = dict()

    assert isinstance(model, torch.nn.Module), "invalid model"

    weight_idx = 0
    bias_idx = 0
    for idx, (name, p) in enumerate(list(model.named_parameters())):
        if 'bn' in name:
            continue
        if 'weight' in name:
            weight[f"layer_{weight_idx}_weight"] = p
            weight_idx += 1
        if 'bias' in name:
            bias[f"layer_{bias_idx}_bias"] = p
            bias_idx += 1

    show_weight = []
    show_bias = []
    for key in weight:
        show_weight.append(weight[key].data.cpu().numpy().flatten())
    for key in bias:
        show_bias.append(bias[key].data.cpu().numpy().flatten())

    # reuse rate
    module_weights = defaultdict(list)
    old_name = ''
    layer_index = 0
    for i, (name, weight) in enumerate(list(model.named_parameters())):
        if 'bn' in name:
            continue
        if ''.join(old_name.split('.')[:-1]) == ''.join(name.split('.')[:-1]):
            module_weights[''.join(old_name.split('.')[:-1])].extend(weight.data.cpu().numpy().flatten())
        else:
            module_weights[''.join(name.split('.')[:-1])].extend(weight.data.cpu().numpy().flatten())
            layer_index += 1
        old_name = name

    max_len = 0
    total_len = 0
    number_weights = []
    min_per_layer = []
    max_per_layer = []
    for _, weight in module_weights.items():
        max_len = max(len(weight), max_len)
        total_len += len(weight)
        number_weights.append(len(weight))
        min_per_layer.append(min(weight))
        max_per_layer.append(max(weight))
    reuse_rate = max_len/total_len
    print("total num: {}, max num: {}, reuse rate: {:.4f}".format(total_len, max_len, reuse_rate))


    if result:

        files = glob.glob(os.path.join(result, '*.txt'))
        idxs = []
        weight_idxs = []
        bias_idxs = []
        device_number = []  # multi devices
        for txtfile in files:

            with open(txtfile, 'r') as f:
                data = f.readlines()  # 45x1
                tmp = []
                for d in data:
                    tmp.append(eval(
                        re.sub('\]', '', re.sub('\[+', '', d.strip()))
                    ))
                idxs.append(tmp[45:])
                if 'weight' in name:
                    weight_idxs.append(tmp[45:])

                    device_num = os.path.basename(txtfile[0]).split('_')[1]
                    device_number.append('col: '+device_num)

                elif 'bias' in name:
                    bias_idxs.append(tmp[45:])
                else:
                    raise ValueError("name error")

        # for file in files:
        #     with open(file, 'r') as f:
        #         data = f.readlines()  # 45x1
        #         tmp = []
        #         for d in data:
        #             tmp.append(eval(
        #                 re.sub('\]', '', re.sub('\[+', '', d.strip()))
        #             ))
        #         idxs.append(tmp[45:])

        ###
        max_time = max(max(x) for x in idxs)
        median_time = statistics.median(sum(idxs, []))
        avg_time = statistics.mean(sum(idxs, []))

        tdm_time = sum(max(x) for x in idxs)
        power = sum(sum(idxs, []))
        print("before time: {}, median time: {},  avg time: {},  "
              "after tdm time: {}, power: {}".format(max_time,
                                                     median_time,
                                                     avg_time,
                                                     tdm_time,
                                                     power))

if __name__ == '__main__':

    args = get_args()

    # model
    if args.model == 'cnn3':
        model = BasicCNN()
    elif args.model == 'cnn5':
        model = BasicCNN5()
    elif args.model == 'cnn7':
        model = BasicCNN7()

    elif args.model == 'res18':
        model = resnet18(pretrained=False, num_classes=100)
    elif args.model == 'res34':
        model = resnet34(pretrained=False, num_classes=100)
    elif args.model == 'res50tiny':
        model = resnet50tiny(pretrained=False, num_classes=100)

    elif args.model == 'vgg6':
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
    print(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model, args.ckpt, device)

    # show
    result = args.quantize_results
    parameters_analysis(model, args,
                        save_fig='./show',
                        show=True,
                        filename=args.ckpt,
                        result=result
                        )
