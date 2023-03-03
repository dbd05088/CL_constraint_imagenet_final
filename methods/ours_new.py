# When we make a new one, we should inherit the Finetune class.
import logging
import copy
from operator import attrgetter
import time
import datetime
import random
import numpy as np
import torch
import pickle
import math

import torch.nn as nn
import torch.nn.functional as F

from methods.cl_manager import CLManagerBase, MemoryBase
from utils.data_loader import cutmix_data, MultiProcessLoader
from utils import autograd_hacks
logger = logging.getLogger()
# writer = SummaryWriter("tensorboard")


class Ours(CLManagerBase):
    def __init__(
            self, train_datalist, test_datalist, device, **kwargs
    ):
        if kwargs["temp_batchsize"] is None:
            kwargs["temp_batchsize"] = 0
        
        # for ours
        self.T = kwargs["temperature"]
        self.corr_warm_up = kwargs["corr_warm_up"]
        self.target_layer = kwargs["target_layer"]
        self.selected_num = 512
        self.corr_map = {}
        self.count_decay_ratio = kwargs["count_decay_ratio"]
        self.k_coeff = kwargs["k_coeff"]
        
        super().__init__(
            train_datalist, test_datalist, device, **kwargs
        )
        self.past_dist_dict = {}
        self.class_std_list = []
        self.features = None
        self.sample_std_list = []
        self.sma_class_loss = {}
        self.normalized_dict = {}
        self.freeze_idx = []
        self.add_new_class_time = []
        self.ver = kwargs["version"]
        self.avg_prob = kwargs["avg_prob"]
        self.weight_option = kwargs["weight_option"]
        self.weight_method = kwargs["weight_method"]
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.prev_weight_list = None
        self.sigma = kwargs["sigma"]
        self.repeat = kwargs["repeat"]
        self.ema_ratio = kwargs['ema_ratio']
        self.weight_ema_ratio = kwargs["weight_ema_ratio"]
        self.use_batch_cutmix = kwargs["use_batch_cutmix"]
        self.device = device
        self.klass_warmup = kwargs["klass_warmup"]
        self.loss_balancing_option = kwargs["loss_balancing_option"]
        self.grad_cls_score_mavg = {}
        self.corr_map_list = []
        self.sample_count_list = []
        self.labels_list=[]
        self._supported_layers = ['Linear', 'Conv2d']
        self.freeze_warmup = 500
        self.grad_dict = {}

        # for gradient subsampling
        
        # self.grad_mavg_base = {n: torch.zeros_like(p) for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        # self.grad_mavgsq_base = {n: torch.zeros_like(p) for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        # self.grad_mvar_base = {n: torch.zeros_like(p) for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        # self.grad_cls_score_mavg_base = {n: 0 for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        # self.grad_criterion_base = {n: 0 for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        # self.grad_dict_base = {n: [] for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        # self.selected_num = 512
        # print("keys")
        # print(self.grad_mavg_base.keys())
        # self.selected_mask = {}
        # for key in self.grad_mavg_base.keys():
        #     a = self.grad_mavg_base[key].flatten()
        #     selected_indices = torch.randperm(len(a))[:self.selected_num]
        #     self.selected_mask[key] = selected_indices
        #     self.grad_mavg_base[key] = torch.zeros(self.selected_num).to(self.device)
        #     self.grad_mavgsq_base[key] = torch.zeros(self.selected_num).to(self.device)
        #     self.grad_mvar_base[key] = torch.zeros(self.selected_num).to(self.device)
        #
        # print("self.selected_mask.keys()")
        # print(self.selected_mask.keys())

        self.last_grad_mean = 0.0

        self.grad_mavg = []
        self.grad_mavgsq = []
        self.grad_mvar = []
        self.grad_criterion = []
        
        self.grad_ema_ratio = 0.01

        # Information based freezing
        self.unfreeze_rate = kwargs["unfreeze_rate"]
        # Information based freezing
        self.fisher_ema_ratio = 0.01
        if self.model_name == 'resnet18':
            self.num_blocks = 9
            self.fisher = [0.0 for _ in range(9)]
        else:
            raise NotImplementedError("Layer blocks for Fisher Information calculation not defined")
        self.cumulative_fisher = []
        self.frozen = False


        self.cumulative_fisher = []

        self.klass_train_warmup = kwargs["klass_train_warmup"]

        self.recent_ratio = kwargs["recent_ratio"]
        self.gpu_transform = kwargs["gpu_transform"]
        self.use_kornia = kwargs["use_kornia"]
        self.use_amp =  False #kwargs["use_amp"]
        self.cls_weight_decay = kwargs["cls_weight_decay"]
        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for name, p in self.model.named_parameters():
            print(name, p.shape)


    def initialize_future(self):
        self.data_stream = iter(self.train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.cls_dict, self.train_transform, self.data_dir, self.transform_on_gpu, self.cpu_transform, self.device, self.use_kornia, self.transform_on_worker)
        self.memory = OurMemory(self.memory_size, self.T, self.count_decay_ratio, self.k_coeff)

        self.grad_score_per_layer = None

        if self.target_layer == "whole_conv2":
            self.target_layers = ["group1.blocks.block0.conv2.block.0.weight", "group1.blocks.block1.conv2.block.0.weight", "group2.blocks.block0.conv2.block.0.weight", "group2.blocks.block1.conv2.block.0.weight", "group3.blocks.block0.conv2.block.0.weight", "group3.blocks.block1.conv2.block.0.weight", "group4.blocks.block0.conv2.block.0.weight", "group4.blocks.block1.conv2.block.0.weight"]
        elif self.target_layer == "last_conv2":
            self.target_layers = ["group1.blocks.block1.conv2.block.0.weight", "group2.blocks.block1.conv2.block.0.weight", "group3.blocks.block1.conv2.block.0.weight", "group4.blocks.block1.conv2.block.0.weight"]

        autograd_hacks.add_hooks(self.model)
        self.selected_mask = {}
        self.grad_mavg_base = {n: torch.zeros_like(p) for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad and n in self.target_layers}
        for key in self.grad_mavg_base.keys():
            a = self.grad_mavg_base[key].flatten()
            selected_indices = torch.randperm(len(a))[:self.selected_num]
            self.selected_mask[key] = selected_indices
            
        self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.train_count = 0
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.exposed_classes = []
        self.seen = 0
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True

        self.waiting_batch = []
        for i in range(self.future_steps):
            self.load_batch()

    def generate_waiting_batch(self, iterations, similarity_matrix=None):
        for i in range(iterations):
            self.waiting_batch.append(self.temp_future_batch + self.memory.retrieval(self.memory_batch_size, similarity_matrix=similarity_matrix))

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
        if sample["klass"] not in self.memory.cls_list:
            self.memory.add_new_class(sample["klass"])
            self.dataloader.add_new_class(self.memory.cls_dict)
            self.future_add_new_class()
        self.update_memory(sample, self.future_sample_num)
        self.future_num_updates += self.online_iter

        if  self.future_num_updates >= 1:
            if self.future_sample_num >= self.corr_warm_up:
                self.generate_waiting_batch(int(self.future_num_updates), self.corr_map)
            else:
                self.generate_waiting_batch(int(self.future_num_updates))
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0

    def _layer_type(self, layer: nn.Module) -> str:
        return layer.__class__.__name__

    def prev_check(self, idx):
        result = True
        for i in range(idx):
            if i not in self.freeze_idx:
                result = False
                break
        return result

    def unfreeze_layers(self):
        self.frozen = False
        for name, param in self.model.named_parameters():
            param.requires_grad = True

    def freeze_layers(self):
        if len(self.freeze_idx) > 0:
            self.frozen = True
        for i in self.freeze_idx:
            if i==0:
                # freeze initial block
                for name, param in self.model.named_parameters():
                    if "initial" in name:
                        param.requires_grad = False
                continue
            self.freeze_layer((i-1)//2, (i-1)%2)

    def freeze_layer(self, layer_index, block_index=None):
        # group(i)가 들어간 layer 모두 freeze
        if self.target_layer == "last_conv2":
            group_name = "group" + str(layer_index)
        elif self.target_layer == "whole_conv2":
            group_name = "group" + str(layer_index) + ".blocks.block"+str(block_index)

        print("freeze", group_name)
        for name, param in self.model.named_parameters():
            if group_name in name:
                param.requires_grad = False

    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'], sample)
            self.writer.add_scalar(f"train/add_new_class", 1, sample_num)
            self.add_new_class_time.append(sample_num)
            print("seed", self.rnd_seed, "dd_new_class_time")
            print(self.add_new_class_time)
        else:
            self.writer.add_scalar(f"train/add_new_class", 0, sample_num)
        
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train(iterations=int(self.num_updates))
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()


    def future_add_new_class(self):
        ### for calculating similarity ###
        len_key = len(self.corr_map.keys())
        if len_key > 1:
            total_corr = 0.0
            total_corr_count = 0
            for i in range(len_key):
                for j in range(i+1, len_key):
                    total_corr += self.corr_map[i][j]
                    total_corr_count += 1
            self.initial_corr = total_corr / total_corr_count
        else:
            self.initial_corr = None
        
        for i in range(len_key):
            # 모든 class의 avg_corr로 initialize
            self.corr_map[i][len_key] = self.initial_corr
            
        # 자기 자신은 1로 initialize
        self.corr_map[len_key] = {}
        self.corr_map[len_key][len_key] = None


    def add_new_class(self, class_name, sample=None):
        print("!!add_new_class seed", self.rnd_seed)
        self.cls_dict[class_name] = len(self.exposed_classes)
        
        # self.grad_cls_score_mavg[len(self.exposed_classes)] = copy.deepcopy(self.grad_cls_score_mavg_base)
        # self.grad_dict[len(self.exposed_classes)] = copy.deepcopy(self.grad_dict_base)
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        prev_bias = copy.deepcopy(self.model.fc.bias.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
                self.model.fc.bias[:self.num_learned_class - 1] = prev_bias
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model.fc.parameters()})

        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

        autograd_hacks.remove_hooks(self.model)
        autograd_hacks.add_hooks(self.model)
        
        # for unfreezing model

        # initialize with mean
        # if len(self.grad_mavg) >= 2:
        #     self.grad_mavg_base = {key: torch.mean(torch.stack([self.grad_mavg[kls][key] for kls in range(len(self.grad_mavg))]), dim=0) for key in self.grad_mavg_base.keys()}
        #     self.grad_mavgsq_base = {key: torch.mean(torch.stack([self.grad_mavgsq[kls][key] for kls in range(len(self.grad_mavg))]), dim=0) for key in self.grad_mavgsq_base.keys()}
        #     self.grad_mvar_base = {key: torch.mean(torch.stack([self.grad_mvar[kls][key] for kls in range(len(self.grad_mavg))]), dim=0) for key in self.grad_mvar_base.keys()}
        #
        # self.grad_mavg.append(copy.deepcopy(self.grad_mavg_base))
        # self.grad_mavgsq.append(copy.deepcopy(self.grad_mavgsq_base))
        # self.grad_mvar.append(copy.deepcopy(self.grad_mvar_base))
        # self.grad_criterion.append(copy.deepcopy(self.grad_criterion_base))
        
        
        # ### update similarity map ###
        # len_key = len(self.corr_map.keys())
        # if len_key > 1:
        #     total_corr = 0.0
        #     total_corr_count = 0
        #     for i in range(len_key):
        #         for j in range(i+1, len_key):
        #             total_corr += self.corr_map[i][j]
        #             total_corr_count += 1
        #     self.initial_corr = total_corr / total_corr_count
        # else:
        #     self.initial_corr = None
        #
        # for i in range(len_key):
        #     # 모든 class의 avg_corr로 initialize
        #     self.corr_map[i][len_key] = self.initial_corr
        #
        # # 자기 자신은 1로 initialize
        # self.corr_map[len_key] = {}
        # self.corr_map[len_key][len_key] = None
        #print("self.corr_map")
        #print(self.corr_map)

    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            x = data["image"].to(self.device)
            y = data["label"].to(self.device)
            print("y")
            print(y)
            #self.before_model_update()
            self.optimizer.zero_grad()

            logit, loss = self.model_forward(x, y)

            if self.train_count > 2:
                self.get_freeze_idx(logit.detach(), y)
                if np.random.rand() > self.unfreeze_rate:
                    self.freeze_layers()

            _, preds = logit.topk(self.topk, 1, True, True)
            
            loss.backward()
            autograd_hacks.compute_grad1(self.model)
            
            self.optimizer.step()
            #self.update_gradstat(self.sample_num, y)
            
            self.update_correlation(y)

            if not self.frozen:
                self.calculate_fisher()

            autograd_hacks.clear_backprops(self.model)
            
            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
            
            if len(self.freeze_idx) == 0:    
                # forward와 backward가 full로 일어날 때
                self.total_flops += (len(y) * (self.forward_flops + self.backward_flops))
            else:
                self.total_flops += (len(y) * (self.forward_flops + self.get_backward_flops()))
                
            # print("total_flops", self.total_flops)
            # self.writer.add_scalar(f"train/total_flops", self.total_flops, self.sample_num)

            self.unfreeze_layers()
            self.freeze_idx = []
            self.after_model_update()

        print("self.corr_map")
        print(self.corr_map)

        return total_loss / iterations, correct / num_data

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        self.corr_map_list.append(copy.deepcopy(self.corr_map))
        self.sample_count_list.append(copy.deepcopy(self.memory.usage_count))
        self.labels_list.append(copy.deepcopy(self.memory.labels))
        
        # store한 애들 저장
        corr_map_name = "corr_map_list_T_" + str(self.T) + "_decay_" + str(self.k_coeff) + ".pickle"
        sample_count_name = "sample_count_list_T_" + str(self.T) + "_decay_" + str(self.k_coeff) + ".pickle"
        labels_list_name = "labels_list_T_" + str(self.T) + "_decay_" + str(self.k_coeff) + ".pickle"
        
        print("corr_map_name", corr_map_name)
        print("sample_count_name", sample_count_name)
        print("labels_list_name", labels_list_name)
        
        with open(corr_map_name, 'wb') as f:
            pickle.dump(self.corr_map_list, f, pickle.HIGHEST_PROTOCOL)
        
        with open(sample_count_name, 'wb') as f:
            pickle.dump(self.sample_count_list, f, pickle.HIGHEST_PROTOCOL)
        
        with open(labels_list_name, 'wb') as f:
            pickle.dump(self.labels_list, f, pickle.HIGHEST_PROTOCOL)
        
        return super().online_evaluate(test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time)


    def after_model_update(self):
        self.train_count += 1

    def get_backward_flops(self):
        backward_flops = self.backward_flops
        if self.frozen:
            for i in self.freeze_idx:
                backward_flops -= self.comp_backward_flops[i]
        return backward_flops

    def model_forward(self, x, y):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b) # 4
                self.total_flops += (len(logit) * 4) / 10e9
        else:
            with torch.cuda.amp.autocast(self.use_amp):
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.total_flops += (len(logit) * 2) / 10e9
        return logit, loss

    def update_memory(self, sample, sample_num):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            cand_idx = self.memory.cls_idx[cls_to_replace]
            idx_to_replace = random.choice(cand_idx)
            self.memory.replace_sample(sample, sample_num, idx_to_replace)
        else:
            self.memory.replace_sample(sample, sample_num)

    def update_correlation(self, labels):
        cor_dic = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad is True and p.grad is not None and n in self.target_layers[-1]:
                if not p.grad.isnan().any():
                    for i, y in enumerate(labels):
                        sub_sampled = p.grad1[i].clone().detach().clamp(-1000, 1000).flatten()[self.selected_mask[n]]
                        if y.item() not in cor_dic.keys():
                            cor_dic[y.item()] = [sub_sampled]
                        else:
                            cor_dic[y.item()].append(sub_sampled)

        centered_list = []
        key_list = list(cor_dic.keys())

        for key in key_list:
            #print("key", key, "len", len(cor_dic[key]))
            stacked_tensor = torch.stack(cor_dic[key])
            #print("stacked_tensor", stacked_tensor.shape)
            #stacked_tensor -= torch.mean(stacked_tensor, dim=0) # make zero mean
            norm_tensor = torch.norm(stacked_tensor, p=2, dim=1) # make unit vector
            
            for i in range(len(norm_tensor)):
                stacked_tensor[i] /= norm_tensor[i]
                
            #stacked_tensor.div(norm_tensor.expand_as(stacked_tensor))
            '''
            for i in range(len(norm_tensor)):
                stacked_tensor[i] /= norm_tensor[i]
            '''
            centered_list.append(stacked_tensor)
        for i, key_i in enumerate(key_list):
            for j, key_j in enumerate(key_list):
                if key_i > key_j:
                    continue           
                cor_i_j = torch.mean(torch.matmul(centered_list[i], centered_list[j].T)).item()
                # [i][j] correlation update
                '''
                print("key_i", key_i, "key_j", key_j, "cor_i_j", cor_i_j)
                print("self.corr_map[key_i][key_j]")
                print(self.corr_map[key_i][key_j])
                '''
                if self.corr_map[key_i][key_j] is None:
                    if not math.isnan(cor_i_j):
                        self.corr_map[key_i][key_j] = cor_i_j
                else:
                    self.corr_map[key_i][key_j] += self.grad_ema_ratio * (cor_i_j - self.corr_map[key_i][key_j])
        #print("self.corr_map")
        #print(self.corr_map)


    def update_gradstat(self, sample_num, labels):
        for n, p in self.model.named_parameters():
            if n in self.grad_mavg[0]:
                if p.requires_grad is True and p.grad is not None:
                    if not p.grad.isnan().any():
                        for i, y in enumerate(labels):
                            ### use sub-sampled gradient ###
                            sub_sampled = p.grad1[i].clone().detach().clamp(-1000, 1000).flatten()[self.selected_mask[n]]
                            #self.grad_dict[y.item()][n].append(sub_sampled)

                            self.grad_mavg[y.item()][n] += self.grad_ema_ratio * (sub_sampled - self.grad_mavg[y.item()][n])
                            self.grad_mavgsq[y.item()][n] += self.grad_ema_ratio * (sub_sampled ** 2 - self.grad_mavgsq[y.item()][n])
                            self.grad_mvar[y.item()][n] = self.grad_mavgsq[y.item()][n] - self.grad_mavg[y.item()][n] ** 2
                            self.grad_criterion[y.item()][n] = (
                                        torch.abs(self.grad_mavg[y.item()][n]) / (torch.sqrt(self.grad_mvar[y.item()][n]) + 1e-10)).mean().item() 
                            self.grad_cls_score_mavg[y.item()][n] += self.grad_ema_ratio * (self.grad_criterion[y.item()][n] - self.grad_cls_score_mavg[y.item()][n])
       
        for cls, dic in enumerate(self.grad_criterion):
            self.writer.add_scalars("grad_criterion"+str(cls), dic, sample_num)

        # just avg_mean score
        label_count = torch.zeros(len(self.exposed_classes)).to(self.device)
        total_label_count = len(labels)
        for label in labels:
            label_count[label.item()] += 1
        label_ratio = label_count / total_label_count

        ### current scoring 방식 ### 
        self.grad_score_per_layer = {layer: torch.sum(torch.Tensor([self.grad_criterion[klass][layer] for klass in range(len(self.exposed_classes))]).to(self.device) * label_ratio).item() for layer in list(self.grad_criterion[0].keys())}
        self.writer.add_scalars("layer_score", self.grad_score_per_layer, sample_num)


    def calculate_covariance(self):
        last_key = list(self.grad_dict[0].keys())[-1]
        tensor_list = []
        for cls in range(len(self.exposed_classes)):
            tensor_list.append(torch.mean(torch.stack(self.grad_dict[cls][last_key]), dim=0))
        tensor_list = torch.stack(tensor_list)
        corr_coeff = torch.corrcoef(tensor_list)
        print(corr_coeff)

    def get_layer_number(self, n):
        name = n.split('.')
        if name[0] == 'initial':
            return 0
        elif 'group' in name[0]:
            group_num = int(name[0][-1])
            block_num = int(name[2][-1])
            return group_num * 2 + block_num - 1

    # Hyunseo : Information based freeezing
    def calculate_fisher(self):
        group_fisher = [0.0 for _ in range(self.num_blocks)]
        for n, p in list(self.model.named_parameters())[:-2]:
            layer_num = self.get_layer_number(n)
            if p.requires_grad is True and p.grad is not None:
                if not p.grad.isnan().any():
                    block_name = '.'.join(n.split('.')[:-3])
                    get_attr = attrgetter(block_name)
                    group_fisher[layer_num] += (p.grad.clone().detach().clamp(-1000, 1000) ** 2).sum().item()/get_attr(self.model).input_scale
                    if self.unfreeze_rate < 1:
                        self.total_flops += (len(p.grad.clone().detach().flatten())*2+get_attr(self.model).input_size*2) / 10e9

        for i in range(self.num_blocks):
            if i not in self.freeze_idx or not self.frozen:
                self.fisher[i] += self.fisher_ema_ratio * (group_fisher[i] - self.fisher[i])
        self.total_fisher = sum(self.fisher)
        self.cumulative_fisher = [sum(self.fisher[0:i+1]) for i in range(9)]

    def get_flops_parameter(self):
        super().get_flops_parameter()
        self.cumulative_backward_flops = [sum(self.comp_backward_flops[0:i+1]) for i in range(9)]
        self.total_model_flops = self.forward_flops + self.backward_flops

    def get_freeze_idx(self, logit, label):
        grad = self.get_grad(logit, label, self.model.fc.weight)
        last_grad = (grad ** 2 ).sum().item()
        if self.unfreeze_rate < 1:
            self.total_flops += len(grad.clone().detach().flatten())/10e9
        batch_freeze_score = last_grad/(self.last_grad_mean+1e-10)
        self.last_grad_mean += self.fisher_ema_ratio * (last_grad - self.last_grad_mean)
        freeze_score = []
        freeze_score.append(1)
        for i in range(9):
            freeze_score.append(self.total_model_flops / (self.total_model_flops - self.cumulative_backward_flops[i]) * (
                        self.total_fisher - self.cumulative_fisher[i]) / (self.total_fisher + 1e-10))
        max_score = max(freeze_score)
        modified_score = []
        modified_score.append(batch_freeze_score)
        for i in range(9):
            modified_score.append(batch_freeze_score*(self.total_fisher - self.cumulative_fisher[i])/(self.total_fisher + 1e-10) + self.cumulative_backward_flops[i]/self.total_model_flops * max_score)
        optimal_freeze = np.argmax(modified_score)
        print(modified_score, optimal_freeze)
        self.freeze_idx = list(range(9))[0:optimal_freeze]

    def get_grad(self, logit, label, weight):
        prob = F.softmax(logit)
        oh_label = F.one_hot(label.long(), self.num_learned_class)

        front = (prob - oh_label).shape
        back = weight.shape
        if self.unfreeze_rate < 1:
            self.total_flops += ((front[0] * back[1] * (2 * front[1] - 1)) / 10e9)

        return torch.matmul((prob - oh_label), weight)


class OurMemory(MemoryBase):
    def __init__(self, memory_size, T, count_decay_ratio, k_coeff):
        super().__init__(memory_size)
        self.T = T
        self.k_coeff = k_coeff
        self.entered_time = []
        self.count_decay_ratio = count_decay_ratio

    def replace_sample(self, sample, sample_num, idx=None):
        super().replace_sample(sample, idx)
        self.usage_count = np.append(self.usage_count, 0)
        self.entered_time.append(sample_num)
    
    # balanced probability retrieval
    def balanced_retrieval(self, size):
        sample_size = min(size, len(self.images))
        memory_batch = []
        cls_idx = np.random.choice(len(self.cls_list), sample_size)
        for cls in cls_idx:
            i = np.random.choice(self.cls_idx[cls], 1)[0]
            memory_batch.append(self.images[i])
            self.usage_count[i]+=1
            self.class_usage_count[self.labels[i]]+=1
        return memory_batch

    
    def retrieval(self, size, similarity_matrix=None):
        # for use count decaying
        if len(self.images) > size:
            self.count_decay_ratio = size / (len(self.images)*self.k_coeff)  #(self.k_coeff / (len(self.images)*self.count_decay_ratio))
            print("count_decay_ratio", self.count_decay_ratio)
            self.usage_count *= (1-self.count_decay_ratio)
            print("self.usave_count")
            print(self.usage_count)
        
        if similarity_matrix is None:
            return self.balanced_retrieval(size)
        else:
            sample_size = min(size, len(self.images))
            weight = self.get_similarity_weight(similarity_matrix)
            sample_idx = np.random.choice(len(self.images), sample_size, p = weight)
            memory_batch = list(np.array(self.images)[sample_idx])
            for i in sample_idx:
                self.usage_count[i]+=1
                self.class_usage_count[self.labels[i]]+=1    
            return memory_batch
        
    def get_similarity_weight(self, sim_matrix):
        weight = np.array(self.usage_count).astype(np.float64)
        #total_count = sum(self.class_usage_cnt)
        
        for my_klass, my_klass_count in enumerate(self.class_usage_count):
            klass_index = np.where(my_klass == np.array(self.labels))[0]
            x = 0
            for other_klass, other_class_count in enumerate(self.class_usage_count):
                min_klass = min(my_klass, other_klass)
                max_klass = max(my_klass, other_klass)
                if sim_matrix[min_klass][max_klass] is None:
                    continue
                other_klass_index = np.where(other_klass == np.array(self.labels))[0]
                other_class_decayed_count = np.sum(self.usage_count[other_klass_index])
                print("other_class_count", other_class_count, "other_class_decayed_count", other_class_decayed_count)
                x += (sim_matrix[min_klass][max_klass] * other_class_decayed_count)
            weight[klass_index] += x 

        '''
        total_count = sum(weight)
        weight = torch.exp(-(weight/total_count)*self.T).double()
        weight = F.softmax(weight, dim=0)
        '''
        weight = np.exp(-(weight)/self.T)
        weight /= sum(weight)
        return weight
    
    