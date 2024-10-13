
import os.path
from collections import OrderedDict
import time

import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm



class BasicEQModule(object):

    def __init__(self, model,
                 codebook,
                 all_layers=False,
                 device_number=1,
                 **kwargs):
        super(BasicEQModule, self).__init__()

        assert isinstance(model, (nn.Module, OrderedDict)) and \
               isinstance(codebook, torch.Tensor), "invalid args"

        self.codebook = codebook #quantization codebook
        self.Qmodules = dict(nn.Sequential(model._modules).named_parameters())

        # delete bn
        for key in list(self.Qmodules):
            if 'bn' in key:
                self.Qmodules.pop(key)
            if 'downsample.1' in key:
                self.Qmodules.pop(key)

        ## add bn mean and var
        # for name, value in dict(model.named_buffers()).items():
        #     if name.split('.')[-1] in "\t".join(["running_mean", "running_var"]):
        #         self.Qmodules[name] = value

        # whether to do global quantization(consider all layers' weights at one time) or a layered method
        self.all_layers = all_layers
        self.device_number = device_number

    def _align(self, source: torch.Tensor, target: torch.Tensor, module_name=''):

        """
        uniform alignment
        """

        target_max, target_min = torch.max(target), torch.min(target)
        source_max, source_min = torch.max(source), torch.min(source)
        S = (source_max - source_min) / (target_max - target_min)
        P = (source - source_min) / S + target_min

        return P

    def _scaling(self, R, codebook, module_name=''):
        """
        TODO:
        # different methods: (min, max) w->device; device->w; and
        # optimal device->w

        :return:
        """

        print("quantizing module: {} ...".format(module_name))
        codebook = self._align(codebook, R.reshape(-1,1), module_name)

        module = self.replace(R, codebook, module_name)

        return module, module_name

    def replace(self, R, P, module_name=''):
        """
        override parameters: nearest quantization
        :return:
        """
        # todo: torch.abs & multi-codebooks

        index = torch.argmin(torch.abs(R.reshape(-1, 1).expand(-1,P.shape[0]) -
                                       P.permute(1,0).expand(R.reshape(-1,1).shape[0],-1)), dim=1)

        if not self.all_layers:
            weight = P[index].reshape(self.Qmodules[module_name].data.shape)
            return weight

        else:
            number = 0
            for m in self.Qmodules:
                length = torch.prod(torch.tensor(self.Qmodules[m].data.shape))
                self.Qmodules[m].data = \
                    P[index][number:number+length].reshape(self.Qmodules[m].data.shape)
                number += length


    def quantize_inference(self):

        R = []
        names = []

        start = time.time()
        with tqdm(total=len(self.Qmodules)-1) as pbar:
            for m in self.Qmodules:
                weight = self.Qmodules[m].data
                if self.all_layers:
                    R.extend(weight.flatten())
                else:
                    new_weight, name = self._scaling(weight, self.codebook, module_name=m)
                    self.Qmodules[name].data = new_weight
                pbar.update(1)
        #
        print(time.time() - start)


        if self.all_layers:
            R = torch.tensor(R).to(device=weight.device)
            self._scaling(R, self.codebook)
        else:
            pass

            # start = time.time()
            # multi_pool_args = [(weight, self.codebook, name) for weight, name in zip(R, names)]
            # new_modules = {}
            # with Pool(processes=8) as pool:
            #     results = pool.starmap_async(self._scaling, multi_pool_args)
            #     for weight, name in results.get():
            #         # new_modules[name] = weight
            #         self.Qmodules[name].data = weight
            # print(time.time() - start)
            # # for m in self.Qmodules:
            # #     self.Qmodules[m].data = new_modules[m]
            # del new_modules

        return self.Qmodules

# the standard one at experiments
class OptimalEQModule(BasicEQModule):
    def __init__(self, *args,
                 initial_temperature=100,
                 cooling_rate=0.99,
                 stopping_temperature=1e-8,
                 search_range=0.001,
                 waiting_steps=100,
                 show_scale='./show/cifar/device_scale/randn_init/res18', # Save the codebook index value corresponding to model parameters
                 **kwargs):
        super(OptimalEQModule, self).__init__(*args,**kwargs)

        self.initial_temp = initial_temperature
        self.cooling_rate = cooling_rate
        self.stopping_temp = stopping_temperature
        self.search_range = search_range
        self.waiting_steps = waiting_steps
        self.show_scale = show_scale

        if show_scale:
            if not os.path.exists(show_scale):
                os.makedirs(show_scale)

    def _cost(self, R, P):
        index = torch.argmin(torch.abs(R.reshape(-1, 1).expand(-1,P.shape[0]) -
                                       P.permute(1,0).expand(R.reshape(-1,1).shape[0],-1)), dim=1)

        # todo: mse?
        return torch.sum(torch.pow(R - P[index], 2))

    def _find_factories(self, source, scale, offset, target):


        # todo: find the optimal scale correlations
        # optimization

        # todo: how to decide the optimal cors? how to choice the `distance` or metric?

        current_scale = scale
        current_offset = offset
        scaled_source = torch.mul(current_scale, source) + current_offset
        current_cost = self._cost(target, scaled_source)
        current_temp = self.initial_temp
        best_scale = current_scale
        best_offset = current_offset
        best_cost = current_cost

        epoch = 0

        while current_temp > self.stopping_temp:

            for i in range(self.waiting_steps):

                # new solution
                new_scale = current_scale + torch.rand_like(current_scale) * 2 * self.search_range - self.search_range
                new_offset = current_offset + torch.rand_like(current_offset) * 2 * self.search_range - self.search_range

                scaled_source = torch.mul(new_scale, source) + new_offset
                new_cost = self._cost(target, scaled_source)

                # acceptance probability
                delta_cost = new_cost - current_cost
                if delta_cost > 0:
                    acceptance_probability = torch.exp(-delta_cost / current_temp).data.cpu()
                else:
                    acceptance_probability = torch.ones(1)

                # update
                if acceptance_probability > torch.rand(1):
                    current_scale = new_scale
                    current_offset = new_offset
                    current_cost = new_cost

                    if current_cost < best_cost:
                        best_scale = current_scale
                        best_offset = current_offset
                        best_cost = current_cost


                if epoch % 20 == 0 and i % 50 == 0:
                    print("temp: {} || step: {} || best cost: {}".format(current_temp,
                                                                     i,
                                                                     best_cost))
                    # print(current_scale, current_offset)
            # cool
            current_temp *= self.cooling_rate
            epoch += 1

        best_scaled_source = torch.mul(best_scale, source) + best_offset

        return (best_scale, best_offset), best_scaled_source

    def _align(self, source: torch.Tensor, target: torch.Tensor, module_name=''):

        """
        optimal align
        """

        #

        scale = torch.randn(1,1).to(device=source.device)
        offset = torch.randn(1,1).to(device=source.device)

        ## different init methods

        # scale = torch.rand((1,1), device=source.device) * 2 - 1
        # offset = torch.rand((1,1), device=source.device) * 2 - 1

        factories, P = self._find_factories(source, scale, offset, target)

        if self.show_scale:
            self.show_scale_curve(*factories, P, module_name)

        return P

    def show_scale_curve(self, scale, offset, data, module_name=''):

        # Save the codebook index value corresponding to model parameters
        file = os.path.join(self.show_scale,
                            f'Device_{self.device_number}_Module_{module_name}.txt')

        P = self.Qmodules[module_name].reshape(-1,1)
        index = torch.argmin(torch.abs(P.reshape(-1, 1).expand(-1, data.shape[0]) -
                                       data.permute(1, 0).expand(P.reshape(-1, 1).shape[0], -1)), dim=1)

        show_case = torch.cat((scale, offset, data, index[:,None]), dim=0)
        with open(file, 'w') as f:
            f.write(np.array2string(show_case.data.cpu().numpy(),
                                    formatter={'float_kind': lambda x: "%.4f" % x},
                                    threshold=np.inf))


class BasicEQModuleV2(object):

    def __init__(self, model,
                 codebook,
                 all_layers=False,
                 device_number=1,
                 **kwargs):
        super(BasicEQModuleV2, self).__init__()

        assert isinstance(model, (nn.Module, OrderedDict)) and \
               isinstance(codebook, torch.Tensor), "invalid args"

        self.codebook = codebook #quantization codebook
        self.Qmodules_tmp = dict(nn.Sequential(model._modules).named_parameters())

        # delete bn
        for key in list(self.Qmodules_tmp):
            if 'bn' in key:
                self.Qmodules_tmp.pop(key)
            if 'downsample.1' in key:
                self.Qmodules_tmp.pop(key)

        ## add bn mean and var
        # for name, value in dict(model.named_buffers()).items():
        #     if name.split('.')[-1] in "\t".join(["running_mean", "running_var"]):
        #         self.Qmodules[name] = value

        self.Qmodules = {}

        module_names = list(self.Qmodules_tmp.keys())
        visited_names = []
        length = []
        for name in module_names:
            if name in visited_names:
                continue
            else:
                if 'weight' in name:
                    tmp_data = self.Qmodules_tmp[name].data.view(-1,1)
                    length.append(len(tmp_data))
                    visited_names.append(name)
                if name.rsplit('.',1)[0] + '.bias' in module_names:
                    visited_names.append(name.rsplit('.',1)[0] + '.bias')
                    bias_data = self.Qmodules_tmp[name.rsplit('.',1)[0] + '.bias'].data.view(-1,1)
                    length.append(len(bias_data))
                    tmp_data = torch.cat([tmp_data, bias_data], dim=0)

                self.Qmodules[name.rsplit('.',1)[0]] = {"data": tmp_data,
                                                        "length": length}
                length = []
        # whether to do global quantization(consider all layers' weights at one time) or a layered method
        self.all_layers = all_layers
        self.device_number = device_number

    def _align(self, source: torch.Tensor, target: torch.Tensor, module_name=''):

        """
        uniform alignment
        """

        target_max, target_min = torch.max(target), torch.min(target)
        source_max, source_min = torch.max(source), torch.min(source)
        S = (source_max - source_min) / (target_max - target_min)
        P = (source - source_min) / S + target_min

        return P

    def _scaling(self, R, codebook, module_name=''):
        """
        TODO:
        # different methods: (min, max) w->device; device->w; and
        # optimal device->w

        :return:
        """

        print("quantizing module: {} ...".format(module_name))
        codebook = self._align(codebook, R.reshape(-1,1), module_name)

        module = self.replace(R, codebook, module_name)

        return module, module_name

    def replace(self, R, P, module_name=''):
        """
        override parameters: nearest quantization
        :return:
        """
        # todo: torch.abs & multi-codebooks

        index = torch.argmin(torch.abs(R.reshape(-1, 1).expand(-1,P.shape[0]) -
                                       P.permute(1,0).expand(R.reshape(-1,1).shape[0],-1)), dim=1)

        return P[index]

    def quantize_inference(self):

        start = time.time()
        with tqdm(total=len(self.Qmodules)-1) as pbar:
            for m in self.Qmodules:
                weight = self.Qmodules[m]["data"]
                new_weight, name = self._scaling(weight, self.codebook, module_name=m)
                length = self.Qmodules[m]["length"]
                if len(length) == 1:
                    self.Qmodules[name]["data"] = new_weight
                else:
                    weight_name = name + '.weight'
                    bias_name = name + '.bias'
                    self.Qmodules_tmp[weight_name].data = new_weight[:length[0]].reshape(self.Qmodules_tmp[weight_name].data.shape)
                    self.Qmodules_tmp[bias_name].data = new_weight[length[0]:].reshape(
                        self.Qmodules_tmp[bias_name].data.shape)
                pbar.update(1)

        print(time.time() - start)

        return self.Qmodules


class OptimalEQModuleV2(BasicEQModuleV2):
    def __init__(self, *args,
                 initial_temperature=100,
                 cooling_rate=0.99,
                 stopping_temperature=1e-8,
                 search_range=0.001,
                 waiting_steps=100,
                 show_scale='./show/cifar/device_scale/randn_init/res18',
                 **kwargs):
        super(OptimalEQModuleV2, self).__init__(*args,**kwargs)

        self.initial_temp = initial_temperature
        self.cooling_rate = cooling_rate
        self.stopping_temp = stopping_temperature
        self.search_range = search_range
        self.waiting_steps = waiting_steps
        self.show_scale = show_scale

        if show_scale:
            if not os.path.exists(show_scale):
                os.makedirs(show_scale)

    def replace(self, R, P, module_name=''):
        """
        override parameters: nearest quantization
        :return:
        """
        index = torch.argmin(torch.abs(R.reshape(-1, 1).expand(-1,P.shape[0]) -
                                       P.permute(1,0).expand(R.reshape(-1,1).shape[0],-1)), dim=1)

        return P[index]



    def _cost(self, R, P):
        index = torch.argmin(torch.abs(R.reshape(-1, 1).expand(-1,P.shape[0]) -
                                       P.permute(1,0).expand(R.reshape(-1,1).shape[0],-1)), dim=1)

        return torch.sum(torch.pow(R - P[index], 2))

    def _find_factories(self, source, scale, offset, target):


        # optimization

        current_scale = scale
        current_offset = offset
        scaled_source = torch.mul(current_scale, source) + current_offset
        current_cost = self._cost(target, scaled_source)
        current_temp = self.initial_temp
        best_scale = current_scale
        best_offset = current_offset
        best_cost = current_cost

        epoch = 0

        while current_temp > self.stopping_temp:

            for i in range(self.waiting_steps):

                # new solution
                new_scale = current_scale + torch.rand_like(current_scale) * 2 * self.search_range - self.search_range
                new_offset = current_offset + torch.rand_like(current_offset) * 2 * self.search_range - self.search_range

                scaled_source = torch.mul(new_scale, source) + new_offset
                new_cost = self._cost(target, scaled_source)

                # acceptance probability
                delta_cost = new_cost - current_cost
                if delta_cost > 0:
                    acceptance_probability = torch.exp(-delta_cost / current_temp).data.cpu()
                else:
                    acceptance_probability = torch.ones(1)

                # update
                if acceptance_probability > torch.rand(1):
                    current_scale = new_scale
                    current_offset = new_offset
                    current_cost = new_cost

                    if current_cost < best_cost:
                        best_scale = current_scale
                        best_offset = current_offset
                        best_cost = current_cost


                if epoch % 20 == 0 and i % 50 == 0:
                    print("temp: {} || step: {} || best cost: {}".format(current_temp,
                                                                     i,
                                                                     best_cost))

            current_temp *= self.cooling_rate
            epoch += 1

        best_scaled_source = torch.mul(best_scale, source) + best_offset

        return (best_scale, best_offset), best_scaled_source

    def _align(self, source: torch.Tensor, target: torch.Tensor, module_name=''):

        """
        optimal align
        """

        scale = torch.randn(1,1).to(device=source.device)
        offset = torch.randn(1,1).to(device=source.device)
        factories, P = self._find_factories(source, scale, offset, target)

        if self.show_scale:
            self.show_scale_curve(*factories, P, module_name)

        return P

    def show_scale_curve(self, scale, offset, data, module_name=''):

        P = self.Qmodules[module_name]["data"].reshape(-1,1)
        index = torch.argmin(torch.abs(P.reshape(-1, 1).expand(-1, data.shape[0]) -
                                       data.permute(1, 0).expand(P.reshape(-1, 1).shape[0], -1)), dim=1)

        if len(self.Qmodules[module_name]["length"]) != 1:
            assert len(self.Qmodules[module_name]["length"]) == 2, "error"
            start = 0
            for i, length in enumerate(self.Qmodules[module_name]["length"]):
                if i == 0:
                    file = os.path.join(self.show_scale,
                                        f'Device_{self.device_number}_Module_{module_name}.weight.txt')
                    show_case = torch.cat((scale, offset, data, index[:,None][start:length]), dim=0)
                    with open(file, 'w') as f:
                        f.write(np.array2string(show_case.data.cpu().numpy(),
                                        formatter={'float_kind': lambda x: "%.4f" % x},
                                        threshold=np.inf))
                if i == 1:
                    file = os.path.join(self.show_scale,
                                        f'Device_{self.device_number}_Module_{module_name}.bias.txt')
                    show_case = torch.cat((scale, offset, data, index[:, None][self.Qmodules[module_name]["length"][0]:]), dim=0)
                    with open(file, 'w') as f:
                        f.write(np.array2string(show_case.data.cpu().numpy(),
                                                formatter={'float_kind': lambda x: "%.4f" % x},
                                                threshold=np.inf))

        if not os.path.exists(os.path.join(self.show_scale+"_merged_weight_bias")):
            os.makedirs(os.path.join(self.show_scale+"_merged_weight_bias"))
        file = os.path.join(self.show_scale+"_merged_weight_bias",
                            f'Device_{self.device_number}_Module_{module_name}.txt')
        show_case = torch.cat((scale, offset, data, index[:, None]), dim=0)
        with open(file, 'w') as f:
            f.write(np.array2string(show_case.data.cpu().numpy(),
                                    formatter={'float_kind': lambda x: "%.4f" % x},
                                    threshold=np.inf))