# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import time
import datetime
import random
import numpy as np
import pandas as pd
import torch
import pickle
import math
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
import torch.nn as nn
from utils.cka import linear_CKA
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from scipy.stats import chi2, norm
#from ptflops import get_model_complexity_info
from flops_counter.ptflops import get_model_complexity_info
from collections import Counter
from methods.er_baseline import ER
from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, cutmix_data_two, get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler
from utils.focal_loss import FocalLoss
logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

class Ours(ER):
    def __init__(
            self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.n_classes = n_classes
        self.exposed_classes = []
        self.seen = 0
        self.topk = kwargs["topk"]
        self.past_dist_dict = {}
        self.class_std_list = []
        self.features = None
        self.sample_std_list = []
        self.sma_class_loss = {}
        self.freeze_idx = []
        self.ver = kwargs["version"]
        self.avg_prob = kwargs["avg_prob"]
        self.weight_option = kwargs["weight_option"]
        self.weight_method = kwargs["weight_method"]
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.prev_weight_list = None
        self.sigma = kwargs["sigma"]
        self.threshold = kwargs["threshold"]
        self.unfreeze_threshold = kwargs["unfreeze_threshold"]
        self.repeat = kwargs["repeat"]
        self.ema_ratio = kwargs['ema_ratio']
        self.weight_ema_ratio = kwargs["weight_ema_ratio"]
        self.use_batch_cutmix = kwargs["use_batch_cutmix"]
        self.device = device
        self.line_fitter = LinearRegression()
        self.klass_warmup = kwargs["klass_warmup"]
        self.loss_balancing_option = kwargs["loss_balancing_option"]
        self.use_class_balancing = kwargs["use_class_balancing"]
        print("self.use_class_balancing", self.use_class_balancing)
        self.dataset = kwargs["dataset"]
        self.model_name = kwargs["model_name"]
        self.opt_name = kwargs["opt_name"]
        self.sched_name = kwargs["sched_name"]
        if self.sched_name == "default":
            self.sched_name = 'const'
        self.lr = kwargs["lr"]

        self.train_transform = train_transform
        self.cutmix = "cutmix" in kwargs["transforms"]
        self.test_transform = test_transform

        self.klass_train_warmup = kwargs["klass_train_warmup"]
        self.memory_size = kwargs["memory_size"]
        self.interval = kwargs["interval"]
        self.data_dir = kwargs["data_dir"]
        #self.use_human_training = kwargs["use_human_training"]
        self.use_human_training = False
        self.curriculum_option = kwargs["curriculum_option"]
        self.online_iter = kwargs["online_iter"]
        self.batch_size = kwargs["batchsize"]
        self.temp_batchsize = kwargs["temp_batchsize"]
        if self.temp_batchsize is None:
            self.temp_batchsize = self.batch_size//2
        if self.temp_batchsize > self.batch_size:
            self.temp_batchsize = self.batch_size
        self.memory_size -= self.temp_batchsize

        self.recent_ratio = kwargs["recent_ratio"]
        self.gpu_transform = kwargs["gpu_transform"]
        self.use_kornia = kwargs["use_kornia"]
        self.use_amp = kwargs["use_amp"]
        self.cls_weight_decay = kwargs["cls_weight_decay"]

        
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.model = select_model(self.model_name, self.dataset, 1).to(self.device)
        self.ema_model= copy.deepcopy(self.model).to(self.device)
        
        for name, p in self.model.named_parameters():
    	    print(name, p.shape)
        
        # for reference model
        '''
        self.reference_model = copy.deepcopy(self.model)
        reference_model_path = str(self.dataset) + "_reference_model_state_dict.pth"
        state_dict = torch.load(reference_model_path)
        fc_key = [key for key in state_dict.keys() if 'fc' in key]
        for key in fc_key:
            state_dict[key] = copy.deepcopy(self.model.state_dict()[key])
        self.reference_model.load_state_dict(state_dict)
        self.reference_model = self.reference_model.to(self.device)
        self.reference_model.eval()
        '''
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.99995
        else:
            self.lr_gamma = 0.9999
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

        self.criterion = criterion.to(self.device)
        
        '''
        criterion_kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        self.criterion = FocalLoss(**criterion_kwargs)
        '''
        self.memory = MemoryDataset(self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, use_kornia=self.use_kornia, 
                                    cls_weight_decay = self.cls_weight_decay, weight_option = self.weight_option, 
                                    weight_ema_ratio = self.weight_ema_ratio, use_human_training=self.use_human_training, 
                                    klass_warmup = self.klass_warmup, klass_train_warmup = self.klass_train_warmup)
        self.temp_batch = []
        self.num_updates = 0
        self.train_count = 0
        self.batch_size = kwargs["batchsize"]
        self.total_count = 0
        self.gt_label = None
        self.test_records = []
        self.n_model_cls = []

        # for valid set
        self.valid_list = []
        self.valid_size = round(self.memory_size * 0.01)
        self.memory_size = self.memory_size - self.valid_size
        self.val_per_cls = self.valid_size
        self.val_full = False

        self.forgetting = []
        self.knowledge_gain = []
        self.total_knowledge = []
        self.retained_knowledge = []
        self.forgetting_time = []
        self.note = kwargs['note']
        self.rnd_seed = kwargs['rnd_seed']
        self.save_path = f'results/{self.dataset}/{self.note}/seed_{self.rnd_seed}'
        self.f_calculated = False
        self.total_flops = 0.0
        self.f_period = kwargs['f_period']
        self.f_next_time = 0
        self.start_time = time.time()
        num_samples = {'cifar10': 50000, 'cifar100': 50000, 'tinyimagenet': 100000, 'imagenet': 1281167}
        self.total_samples = num_samples[self.dataset]

    def update_sma_class_loss(self, cls_loss):

        for cls, loss in enumerate(cls_loss):
            if len(self.sma_class_loss[cls]) <= 10:
                if not math.isnan(loss):
                    self.sma_class_loss[cls].append(loss)
            else:
                self.sma_class_loss[cls].pop(0)
                self.sma_class_loss[cls].append(loss)

        #print("sma_class_loss")
        #print(self.sma_class_loss)

    def get_total_flops(self):
        return self.total_flops

    def online_validate(self, sample_num, batch_size, n_worker):
        # for validation
        val_df = pd.DataFrame(self.valid_list)
        print()
        print("### val_df ###")
        print(len(val_df))
        print(Counter(list(val_df['klass'].values)))
        print()
        exp_val_df = val_df[val_df['klass'].isin(self.exposed_classes)]
        val_dataset = ImageDataset(
            exp_val_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        val_dict = self.validation(val_loader, nn.CrossEntropyLoss(reduction="none"), sample_num, per_class_loss=True)

        # check for layer freezing
        interval = self.interval
        threshold = self.threshold #1e-4
        unfreeze_threshold = self.unfreeze_threshold #1e-4
        
        coeff_dict = {}
        
        for idx, key in enumerate(list(self.past_dist_dict.keys())):
            distances = self.past_dist_dict[key]
            
            if not self.prev_check(idx):
                # 앞의 layer가 다 freeze 되어 있어야만 다음 layer 확인 진행하는거
                break
            
            '''
            #unfreeze 하려면 이부분 주석처리
            if idx in self.freeze_idx:
                # 이미 내 layer가 freeze 되어 있다면 더이상 check X
                continue
            '''
            
            if len(distances) >= interval:
                self.line_fitter.fit(np.array((range(interval))).reshape(-1,1), distances[-interval:])
                #print("abs(line_fitter.coef_)", abs(self.line_fitter.coef_))
                
                coeff_dict[key] = abs(self.line_fitter.coef_)
                
                if abs(self.line_fitter.coef_) < threshold:
                    print("!!freeze", idx, "seed", self.rnd_seed)
                    #self.freeze_layer(idx)
                    self.freeze_idx.append(idx)
                
                if unfreeze_threshold is not None: # Unfreeze 할 때    
                    if abs(self.line_fitter.coef_) > unfreeze_threshold and idx in self.freeze_idx:
                        print("!!unfreeze", idx, "seed", self.rnd_seed)
                        #self.freeze_layer(idx)
                        self.freeze_idx.remove(idx)
            
        
        self.writer.add_scalars(f"val/coeff", coeff_dict, sample_num)
        
        # validation set에서 class_loss
        class_loss = val_dict['cls_loss']
        class_acc = val_dict['my_cls_acc']

        if self.curriculum_option == "class_loss":
            self.update_sma_class_loss(class_loss)
            self.memory.transform_gpu.set_cls_magnitude(self.curriculum_option, class_loss, self.memory.cls_count)
        elif self.curriculum_option == "class_acc":
            self.update_sma_class_loss(class_acc.cpu().numpy())
            self.memory.transform_gpu.set_cls_magnitude(self.curriculum_option, class_acc, self.memory.cls_count)
        else:
            pass
        
        self.report_val(sample_num, val_dict["avg_loss"], val_dict["avg_acc"], class_loss, self.memory.transform_gpu.get_cls_magnitude())

    def prev_check(self, idx):
        result = True
        for i in range(idx):
            if i not in self.freeze_idx:
                result = False
                break
        return result

    def freeze_layers(self):
        for i in self.freeze_idx:
            self.freeze_layer(i)

    def freeze_layer(self, layer_index):
        if layer_index == 0:
            # initial이 들어간 layer 모두 freeze
            for name, param in self.model.named_parameters():
                if "initial" in name:
                    param.requires_grad = False
        else:
            # group(i)가 들어간 layer 모두 freeze
            group_name = "group" + str(layer_index)
            for name, param in self.model.named_parameters():
                if group_name in name:
                    param.requires_grad = False
        
        '''
        # freeze 여부 check     
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)
        '''

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        '''
        # for validation
        if len(self.valid_list) > 0:
            val_df = pd.DataFrame(self.valid_list)
            print()
            print("### val_df ###")
            print(len(val_df))
            #print(Counter(list(val_df['klass'].values)))
            print()
            exp_val_df = val_df[val_df['klass'].isin(self.exposed_classes)]
            val_dataset = ImageDataset(
                exp_val_df,
                dataset=self.dataset,
                transform=self.test_transform,
                cls_list=self.exposed_classes,
                data_dir=self.data_dir
            )
            val_loader = DataLoader(
                val_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=n_worker,
            )
            val_dict = self.evaluation(val_loader, nn.CrossEntropyLoss(reduction="none"), per_class_loss=True)

            # validation set에서 class_loss
            class_loss =val_dict['cls_loss']
            self.update_sma_class_loss(class_loss)

            self.report_val(sample_num, val_dict["avg_loss"], val_dict["avg_acc"], class_loss)
        '''
        # for testing
        test_df = pd.DataFrame(test_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        eval_dict = self.evaluation(test_loader, self.criterion)
        online_acc = self.calculate_online_acc(eval_dict["cls_acc"], data_time, cls_dict, cls_addition)
        eval_dict["online_acc"] = online_acc
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"], eval_dict["online_acc"])

        if sample_num >= self.f_next_time:
            self.get_forgetting(sample_num, test_list, cls_dict, batch_size, n_worker)
            self.f_next_time += self.f_period
            self.f_calculated = True
        else:
            self.f_calculated = False
        
        return eval_dict

    def online_step(self, sample, sample_num, n_worker):
        
        if sample_num == 15000:
            if self.ver == "ver5": # task 1끝나고 freeze layer 1
                #self.freeze_layer(0)
                self.freeze_idx.append(0)
                
            elif self.ver == "ver6": # task 1끝나고 freeze layer 1, 2
                #self.freeze_layer(0)
                #self.freeze_layer(1)
                self.freeze_idx.append(0)
                self.freeze_idx.append(1)
            
            elif self.ver == "ver7": # task 1끝나고 freeze layer 1, 2
                #self.freeze_layer(0)
                #self.freeze_layer(1)
                #self.freeze_layer(2)
                self.freeze_idx.append(0)
                self.freeze_idx.append(1)
                self.freeze_idx.append(2)
            
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'], sample)
            self.writer.add_scalar(f"train/add_new_class", 1, sample_num)
        else:
            self.writer.add_scalar(f"train/add_new_class", 0, sample_num)

        self.total_count += 1
        use_sample = self.online_valid_update(sample)
        if use_sample:
            self.update_memory(sample, self.total_count)

        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            if len(self.memory)>0:
                train_loss, train_acc, initial_CKA, group1_CKA, group2_CKA, group3_CKA, group4_CKA= self.online_train([], self.batch_size, n_worker,
                                                        iterations=int(self.num_updates), stream_batch_size=0, sample_num=sample_num)
                self.report_training(sample_num, train_loss, train_acc) #, initial_CKA, group1_CKA, group2_CKA, group3_CKA, group4_CKA)
                self.num_updates -= int(self.num_updates)
                self.update_schedule()

    def online_valid_update(self, sample):
        val_df = pd.DataFrame(self.valid_list, columns=['klass', 'file_name', 'label'])
        if not self.val_full:
            # memory_size의 0.01배만큼의 validation size를 갖고 있기 때문에, 확률을 대략 0.02를 곱해준 것
            do_val = np.random.rand(1) < 0.5
            if len(val_df[val_df["klass"] == sample["klass"]]) < self.val_per_cls and do_val:
                self.valid_list.append(sample)
                if len(self.valid_list) == self.val_per_cls*self.num_learned_class:
                    self.val_full = True
                use_sample = False
            else:
                use_sample = True
        else:
            use_sample = True
        return use_sample

    '''
    # offline version valid_update
    def online_valid_update(self, sample):
        val_df = pd.DataFrame(self.valid_list, columns=['klass', 'file_name', 'label'])
        if not self.val_full:
            if len(val_df[val_df["klass"] == sample["klass"]]) < self.val_per_cls:
                self.valid_list.append(sample)
                if len(self.valid_list) == self.val_per_cls*self.num_learned_class:
                    self.val_full = True
                use_sample = False
            else:
                use_sample = True
        else:
            use_sample = True
        return use_sample
    '''

    def online_reduce_valid(self, num_learned_class):
        self.val_per_cls = self.valid_size//num_learned_class
        val_df = pd.DataFrame(self.valid_list)
        valid_list = []
        new_index = []
        for klass in val_df["klass"].unique():
            class_val = val_df[val_df.klass == klass]
            if len(class_val) > self.val_per_cls:
                new_class_val = class_val.sample(n=self.val_per_cls)
            else:
                new_class_val = class_val
                
            new_index.extend(list(new_class_val.index))
            valid_list += new_class_val.to_dict(orient="records")

        
        if self.ver == "ver2":
            ############################
            # for ver2
            if self.features is not None:
                new_index = [i for i in new_index if i<len(self.features["block1"])]
                print("new_index")
                print(new_index)
                print("length")
                print(len(self.features["block1"]))
                for i in range(5):
                    block_name = "block"+str(i)
                    print("block_name", block_name)
                    print("self.features[block_name]", self.features[block_name].shape)
                    self.features[block_name] = copy.deepcopy(self.features[block_name][new_index])
                    #drop_index = list(set(class_val.index) - set(new_class_val.index))
            ############################
        
        elif self.ver == "ver4" or self.ver == "ver4_1":
            ############################
            # for ver4
            if self.features is not None:
                # 0번째가 가장 feature length가 작을 것
                new_index = [i for i in new_index if i<len(self.features["block1"][0])]
                for i in range(5):
                    block_name = "block"+str(i)
                    self.features[block_name] = [copy.deepcopy(feature[new_index]) for feature in self.features[block_name]]
            ############################
        
        self.valid_list = valid_list
        self.val_full = False

    @torch.no_grad()
    def update_ema_model(self):
        model_params = OrderedDict(self.model.named_parameters())
        ema_params = OrderedDict(self.ema_model.named_parameters())
        assert model_params.keys() == ema_params.keys()
        for name, param in model_params.items():
            ema_params[name].sub_((1. - self.ema_ratio) * (ema_params[name] - param))
        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.ema_model.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def add_new_class(self, class_name, sample=None):
        print("add_new_class!!")
        self.sma_class_loss[len(self.exposed_classes)] = []
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
        self.memory.add_new_class(cls_list=self.exposed_classes, sample=sample)

        # 처음에는 class 0으로 가득 차있다가 갈수록 class 개수가 증가할수록 1개의 class당 차지하는 공간의 크기를 줄여주는 것
        # 이떄, valid_list에서 제거된 sample은 traning에 쓰이는 것이 아니라, 아예 제거되는 것
        self.ema_model.fc = copy.deepcopy(self.model.fc)
        
        if self.num_learned_class > 1 and len(self.valid_list)>0:
            self.online_reduce_valid(self.num_learned_class)

        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
        
        # for unfreezing model
        '''
        if self.ver not in ["ver5", "ver6", "ver7", "ver8"]:
            self.model.train()
            self.freeze_idx = []
        '''

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=0, sample_num=None):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        initial_CKA, group1_CKA, group2_CKA, group3_CKA, group4_CKA = 0.0, 0.0, 0.0, 0.0, 0.0

        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=True)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

        for i in range(iterations):
            self.model.train()
            self.freeze_layers()

            x = []
            y = []
            x2 = []
            y2 = []
            
            #if len(self.memory) > 0:
            #memory_data = self.memory.get_batch(memory_batch_size, use_weight = True, exp_weight=True, recent_ratio = self.recent_ratio)
            #memory_data = self.memory.get_batch(memory_batch_size, use_weight=False, use_human_training=True)
            #memory_data = self.memory.get_batch(memory_batch_size, use_weight="classwise", weight_method = self.weight_method, n_class = self.num_learned_class, avg_prob = self.avg_prob)
            class_loss = [np.mean(self.sma_class_loss[key]) for key in list(self.sma_class_loss.keys()) if len(self.sma_class_loss[key]) != 0]
            
            if len(class_loss) == 0:
                memory_data = self.memory.get_batch(memory_batch_size, use_weight="classwise", weight_method = self.weight_method)
            else:
                memory_data = self.memory.get_batch(memory_batch_size, use_weight="classwise", weight_method = self.weight_method, class_loss = class_loss)
            
            # std check 위해서
            class_std, sample_std = self.memory.get_std()
            self.class_std_list.append(class_std)
            self.sample_std_list.append(sample_std)
            
            x.append(memory_data['image'])
            y.append(memory_data['label'])
            cls_weight = memory_data['cls_weight']
            counter = memory_data["counter"]

            if self.use_batch_cutmix:
                memory_data2 = self.memory.get_batch(memory_batch_size, prev_batch_index=memory_data['indices'])
                x2.append(memory_data2['image'])
                y2.append(memory_data2['label'])
                x2 = torch.cat(x2)
                y2 = torch.cat(y2)
                x2 = x2.to(self.device)
                y2 = y2.to(self.device)
                    
            x = torch.cat(x)
            y = torch.cat(y)
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            if self.weight_option == "loss":
                if self.use_batch_cutmix:
                    features, logit, loss, cls_loss = self.model_forward(x, y, x2, y2, return_cls_loss=True, loss_balancing_option = self.loss_balancing_option, loss_balancing_weight = cls_weight)
                else:
                    features, logit, loss, cls_loss = self.model_forward(x, y, return_cls_loss=True, loss_balancing_option = self.loss_balancing_option, loss_balancing_weight = cls_weight)
                
                if cls_loss is not None:
                    self.memory.update_sample_loss(memory_data['indices'], cls_loss.detach().cpu())
                    self.memory.update_class_loss(memory_data['indices'], cls_loss.detach().cpu())
            else:
                if self.use_batch_cutmix:
                    features, logit, loss = self.model_forward(x, y, x2, y2, loss_balancing_option = self.loss_balancing_option, loss_balancing_weight = cls_weight)
                else:
                    features, logit, loss = self.model_forward(x, y, loss_balancing_option = self.loss_balancing_option, loss_balancing_weight = cls_weight)


            #_, ref_features = self.reference_model(x, get_features=True)

            '''
            print("features length")
            print(len(features)) # 5
            print("features.shape")
            print(features[0].shape) # [1, 64, 32, 32]
            '''

            '''
            features = [feat.reshape(feat.shape[0], feat.shape[1], feat.shape[2]*feat.shape[3]) for feat in features]
            ref_features = [feat.reshape(feat.shape[0], feat.shape[1], feat.shape[2]*feat.shape[3]) for feat in ref_features]
            initial_CKA += torch.mean(torch.stack([linear_CKA(features[0][i], ref_features[0][i], self.device) for i in range(len(x))]))
            group1_CKA += torch.mean(torch.stack([linear_CKA(features[1][i], ref_features[1][i], self.device) for i in range(len(x))]))
            group2_CKA += torch.mean(torch.stack([linear_CKA(features[2][i], ref_features[2][i], self.device) for i in range(len(x))]))
            group3_CKA += torch.mean(torch.stack([linear_CKA(features[3][i], ref_features[3][i], self.device) for i in range(len(x))]))
            group4_CKA += torch.mean(torch.stack([linear_CKA(features[4][i], ref_features[4][i], self.device) for i in range(len(x))]))
            '''
            '''
            group1_CKA = linear_CKA(features[1], ref_features[1], self.device)
            group2_CKA = linear_CKA(features[2], ref_features[2], self.device)
            group3_CKA = linear_CKA(features[3], ref_features[3], self.device)
            group4_CKA = linear_CKA(features[4], ref_features[4], self.device)
            '''
            _, preds = logit.topk(self.topk, 1, True, True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # update ema model
            self.update_ema_model()
            
            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
            
            if len(self.freeze_idx) == 0:    
                # forward와 backward가 full로 일어날 때
                self.total_flops += (batch_size * (self.forward_flops + self.backward_flops))
            else:
                self.total_flops += (batch_size * (self.forward_flops + self.get_backward_flops()))
                
            print("total_flops", self.total_flops)
            self.writer.add_scalar(f"train/total_flops", self.total_flops, sample_num)

        '''
        initial_CKA = linear_CKA(self.reference_model.initial, self.model.initial)
        group1_CKA = linear_CKA(self.reference_model.group1, self.model.group1)
        group2_CKA = linear_CKA(self.reference_model.group2, self.model.group2)
        group3_CKA = linear_CKA(self.reference_model.group3, self.model.group3)
        group4_CKA = linear_CKA(self.reference_model.group4, self.model.group4)
        fc_CKA = linear_CKA(self.reference_model.fc, self.model.fc)
        '''
        '''
        weight_difference_dict = {}
        
        current_weight_list = []
        for i in range(1, 5):
            for name, p in self.model.named_parameters():
                target_name = "group" + str(i) + ".blocks.block1.conv2.block.1.weight"
                if target_name == name:
                    #print("name", name)
                    current_weight_list.append(p.detach())
        
        if self.prev_weight_list is not None:
            for i in range(0, 4):
                # Euclidian Distance
                #dist = torch.cdist(self.prev_weight_list[i].unsqueeze(0), current_weight_list[i].unsqueeze(0), p=2) # 단순비교
                #dist = torch.cdist(torch.mean(torch.stack(self.prev_weight_list[i]), dim=0).unsqueeze(0), current_weight_list[i].unsqueeze(0), p=2) # moving average
                
                # CKA
                #dist = linear_CKA(torch.mean(torch.stack(self.prev_weight_list[i]), dim=0).unsqueeze(0), current_weight_list[i].unsqueeze(0), self.device)
                #dist = linear_CKA(self.prev_weight_list[i].unsqueeze(0), current_weight_list[i].unsqueeze(0), self.device)
                
                # Cosine Similarity
                #dist = cos(self.prev_weight_list[i], current_weight_list[i])
                #dist = self.cos(torch.mean(torch.stack(self.prev_weight_list[i]), dim=0), current_weight_list[i])
                
                key_name = "block" + str(i)
                weight_difference_dict[key_name] = abs(dist.item())
                print("dist", abs(dist.item()))
                
            
            self.writer.add_scalars(f"train/weight_difference", weight_difference_dict, sample_num)
        
        # 단순 비교
        # self.prev_weight_list = copy.deepcopy(current_weight_list)
        
        # moving average
        self.update_prev_weight(current_weight_list)
        '''
        
        #print('initial CKA: {}, group1 CKA: {}, group2 CKA: {}, group3 CKA: {}, group4 CKA: {}'.format(initial_CKA / iterations, group1_CKA / iterations, group2_CKA / iterations, group3_CKA / iterations, group4_CKA / iterations))

        #return total_loss / iterations, correct / num_data, initial_CKA / iterations, group1_CKA / iterations, group2_CKA / iterations, group3_CKA / iterations, group4_CKA / iterations
        return total_loss / iterations, correct / num_data, None, None, None, None, None

    def get_backward_flops(self):
        backward_flops = self.backward_flops
        
        for i in self.freeze_idx:
            backward_flops -= self.comp_backward_flops[i]
        
        return backward_flops

    def update_prev_weight(self, current_weight_list):
    
        if self.prev_weight_list is None:
            self.prev_weight_list = [] #copy.deepcopy(current_weight_list)
            for idx, weight in enumerate(current_weight_list):
                self.prev_weight_list.append([copy.deepcopy(weight)])
            
        else:
            for cls, weight in enumerate(current_weight_list):
                if len(self.prev_weight_list[cls]) < 5:
                    self.prev_weight_list[cls].append(copy.deepcopy(weight))
                else:
                    self.prev_weight_list[cls].pop(0)
                    self.prev_weight_list[cls].append(copy.deepcopy(weight))
                    

    def model_forward(self, x, y, x2=None, y2=None, return_cls_loss=False, loss_balancing_option=None, loss_balancing_weight=None):
        '''
        self.criterion의 reduction
        1) none
        just element wise 곱셈
        
        2) mean
        element wise 곱셈 후 sum, 후 나눗셈

        3) sum
        just element wise 곱셈 후 sum
        '''
        if loss_balancing_option == "reverse_class_weight":    
            loss_balancing_weight = 1 / np.array(loss_balancing_weight)
            loss_balancing_weight /= np.sum(loss_balancing_weight)
        elif loss_balancing_option == "class_weight": # default
            pass
        else:
            pass

        #do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        #do_cutmix = self.cutmix and np.random.rand(1) < self.memory.get_ratio(y)
        do_cutmix=False

        #TODO Cutmix를 0.5의 확률이 아닌, class별로 magnitude 기반으로 바꿔주기
        
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logit, features = self.model(x, get_features=True)
                    if self.weight_option == "loss": # 이때는 self.criterion(reduction="none")
                        loss_a = self.criterion(logit, labels_a) # 1
                        loss_b = self.criterion(logit, labels_b) # 1
                        total_loss = lam * loss_a + (1 - lam) * (loss_b) # 3
                        
                        if loss_balancing_weight is None:
                            loss = torch.mean(total_loss) # 1
                        else:
                            for i, sample_loss in enumerate(total_loss):
                                if i==0:
                                    loss = loss_balancing_weight[i] * sample_loss
                                else:
                                    loss += loss_balancing_weight[i] * sample_loss
                            
                        self.total_flops += (len(logit) * 6) / 10e9
                    else: # 이때는 self.criterion(reduction="mean")
                        loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b) # 4
                        self.total_flops += (len(logit) * 4) / 10e9
            else:
                logit, features = self.model(x, get_features=True)
                if self.weight_option == "loss":
                    loss_a = self.criterion(logit, labels_a)
                    loss_b = self.criterion(logit, labels_b)
                    total_loss = lam * loss_a + (1 - lam) * (loss_b)
                    loss = torch.mean(total_loss)
                    self.total_flops += (len(logit) * 6) / 10e9
                else:
                    loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
                    self.total_flops += (len(logit) * 4) / 10e9
        else:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logit, features = self.model(x, get_features=True)
                    if self.weight_option == "loss":
                        total_loss = self.criterion(logit, y)
                        loss = torch.mean(total_loss)
                    else:
                        loss = self.criterion(logit, y)
                    self.total_flops += (len(logit) * 2) / 10e9
            else:
                logit, features = self.model(x, get_features=True)
                if self.weight_option == "loss":
                    total_loss = self.criterion(logit, y)

                    if loss_balancing_weight is None:
                        loss = torch.mean(total_loss) # 1
                    else:
                        for i, sample_loss in enumerate(total_loss):
                            if i==0:
                                loss = loss_balancing_weight[i] * sample_loss
                            else:
                                loss += loss_balancing_weight[i] * sample_loss
                else:
                    loss = self.criterion(logit, y)

                self.total_flops += (len(logit) * 2) / 10e9
        if return_cls_loss:
            if do_cutmix:
                return features, logit, loss, None
            else:
                return features, logit, loss, total_loss
        else:
            return features, logit, loss

    def validation(self, test_loader, criterion, sample_num, per_class_loss=False):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        class_loss = torch.zeros(len(self.exposed_classes))
        class_count = torch.zeros(len(self.exposed_classes))
        
        total_class_num = torch.zeros(len(self.exposed_classes))
        correct_class_num = torch.zeros(len(self.exposed_classes))
        
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        
        label = []
        current_feature_dict = {}
        ema_feature_dict = {}
        
        self.model.eval()
        self.ema_model.eval()
        
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit, current_features = self.model(x, get_features=True)
                
                self.total_flops += (len(x) * (self.forward_flops + self.backward_flops))
                
                if self.ver == "ver3":
                    _, ema_features = self.ema_model(x, get_features=True)
                
                for i in range(5):
                    key_name = "block" + str(i)
                    if key_name not in current_feature_dict.keys():
                        current_feature_dict[key_name] = [current_features[i].detach().cpu()]
                        if self.ver == "ver3":
                            ema_feature_dict[key_name] = [ema_features[i].detach().cpu()]
                    else:
                        current_feature_dict[key_name].append(current_features[i].detach().cpu())
                        if self.ver == "ver3":
                            ema_feature_dict[key_name].append(ema_features[i].detach().cpu())
                
                if self.weight_option == "loss" or per_class_loss:
                    loss = criterion(logit, y)
                    if per_class_loss:
                        for i, l in enumerate(y):
                            class_loss[l.item()] += loss[i].item()
                            class_count[l.item()] += 1

                    loss = torch.mean(loss)
                else:
                    loss = criterion(logit, y)
                    
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                
                pred_tensor = preds == y.unsqueeze(1)
                correct_index = (pred_tensor == 1).nonzero(as_tuple=True)[0]
                correct_count = Counter(y[correct_index].cpu().numpy())
                total_class_count = Counter(y.cpu().numpy())
                
                for key in list(correct_count.keys()):
                    correct_class_num[key] += correct_count[key]
                
                for key in list(total_class_count.keys()):
                    total_class_num[key] += total_class_count[key]
                
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()


        if self.ver not in ["ver5", "ver6", "ver7", "ver8"]:
            for i in range(5):
                key_name = "block" + str(i)
                '''
                print("len(current_feature_dict[key_name])", len(current_feature_dict[key_name]))
                for feature in current_feature_dict[key_name]:
                    print("key_name", key_name,"shape", feature.shape)
                '''
                current_feature_dict[key_name] = torch.cat(current_feature_dict[key_name], dim=0)
                if self.ver == "ver3":
                    ema_feature_dict[key_name] = torch.cat(ema_feature_dict[key_name], dim=0)
            
            #print("current", key_name, "shape", current_feature_dict[key_name].shape)
            #print("ema", key_name, "shape", ema_feature_dict[key_name].shape)
        
        
        if self.ver == "ver2":
            ######################
            # for ver2
            if self.features is not None:
                weight_difference_dict = {}
                for i in range(5):
                    key_name = "block" + str(i)
                    dist = abs(((self.features[key_name] - current_feature_dict[key_name][:len(self.features[key_name])])**2).mean().item()) # for ver2
                    weight_difference_dict[key_name] = dist
                    
                    if key_name not in self.past_dist_dict.keys():
                        self.past_dist_dict[key_name] = [dist]
                    else:
                        self.past_dist_dict[key_name].append(dist)
                    
                self.writer.add_scalars(f"train/weight_difference", weight_difference_dict, sample_num)
                
            else:
                self.features = {}
                for key in list(current_feature_dict.keys()):
                    self.features[key] = [copy.deepcopy(current_feature_dict[key])]
                    
                    
            self.features = copy.deepcopy(current_feature_dict)
            ######################
        
        elif self.ver == "ver3":
            ######################
            # for ver3
            weight_difference_dict = {}
            for i in range(5):
                key_name = "block" + str(i)
                dist = abs(((ema_feature_dict[key_name] - current_feature_dict[key_name])**2).mean().item())
                weight_difference_dict[key_name] = dist
                
                if key_name not in self.past_dist_dict.keys():
                    self.past_dist_dict[key_name] = [dist]
                else:
                    self.past_dist_dict[key_name].append(dist)
            self.writer.add_scalars(f"train/weight_difference", weight_difference_dict, sample_num)
            ######################
        
        elif self.ver == "ver4":
            ######################
            # for ver4
            if self.features is not None:
                weight_difference_dict = {}
                for i in range(5):
                    key_name = "block" + str(i)
                    lengths = [len(feature) for feature in self.features[key_name]]
                    stack_candidate = [feature[:min(lengths)] for feature in self.features[key_name]]
                    dist = abs(((torch.mean(torch.stack(stack_candidate), dim=0) - current_feature_dict[key_name][:min(lengths)])**2).mean().item()) # for ver4
                    
                    self.features[key_name].append(copy.deepcopy(current_feature_dict[key_name]))
                    weight_difference_dict[key_name] = dist
                    if key_name not in self.past_dist_dict.keys():
                        self.past_dist_dict[key_name] = [dist]
                    else:
                        self.past_dist_dict[key_name].append(dist)
                    
                self.writer.add_scalars(f"train/weight_difference", weight_difference_dict, sample_num)
                
            else:
                self.features = {}
                for key in list(current_feature_dict.keys()):
                    #self.features[key] = [torch.cat(copy.deepcopy(current_feature_dict[key]), dim=0)]
                    self.features[key] = [copy.deepcopy(current_feature_dict[key])]
            ######################

        elif self.ver == "ver4_1":
            ######################
            # for ver4_1
            if self.features is not None:
                weight_difference_dict = {}
                for i in range(5):
                    key_name = "block" + str(i)
                    lengths = [len(feature) for feature in self.features[key_name]]
                    stack_candidate = [feature[:min(lengths)] for feature in self.features[key_name]]
                    dist = abs(((torch.mean(torch.stack(stack_candidate), dim=0) - current_feature_dict[key_name][:min(lengths)])**2).mean().item()) # for ver4_1
                    
                    self.features[key_name].append(copy.deepcopy(current_feature_dict[key_name]))
                    weight_difference_dict[key_name] = dist
                    if key_name not in self.past_dist_dict.keys():
                        self.past_dist_dict[key_name] = [dist]
                    else:
                        self.past_dist_dict[key_name].append(dist)
                    
                self.writer.add_scalars(f"train/weight_difference", weight_difference_dict, sample_num)
                
            else:
                self.features = {}
                for key in list(current_feature_dict.keys()):
                    #self.features[key] = [torch.cat(copy.deepcopy(current_feature_dict[key]), dim=0)]
                    self.features[key] = [copy.deepcopy(current_feature_dict[key])]
            ######################

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        cls_loss = class_loss.numpy() / class_count.numpy()
        my_cls_acc = correct_class_num / (total_class_num + 0.001)
        
        if not per_class_loss:
            ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        else:
            ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc, "cls_loss" : cls_loss, "my_cls_acc" : my_cls_acc}

        return ret

    def evaluation(self, test_loader, criterion, per_class_loss=False):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        class_loss = torch.zeros(len(self.exposed_classes))
        class_count = torch.zeros(len(self.exposed_classes))
        
        total_class_num = torch.zeros(len(self.exposed_classes))
        correct_class_num = torch.zeros(len(self.exposed_classes))
        
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.model(x)

                if self.weight_option == "loss" or per_class_loss:
                    loss = criterion(logit, y)
                    if per_class_loss:
                        for i, l in enumerate(y):
                            class_loss[l.item()] += loss[i].item()
                            class_count[l.item()] += 1

                    loss = torch.mean(loss)
                else:
                    loss = criterion(logit, y)
                    
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                
                pred_tensor = preds == y.unsqueeze(1)
                correct_index = (pred_tensor == 1).nonzero(as_tuple=True)[0]
                correct_count = Counter(y[correct_index].cpu().numpy())
                total_class_count = Counter(y.cpu().numpy())
                
                for key in list(correct_count.keys()):
                    correct_class_num[key] += correct_count[key]
                
                for key in list(total_class_count.keys()):
                    total_class_num[key] += total_class_count[key]
                
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        
        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        cls_loss = class_loss.numpy() / class_count.numpy()
        
        my_cls_acc = correct_class_num / (total_class_num + 0.001)
        
        
        if not per_class_loss:
            ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        else:
            ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc, "cls_loss" : cls_loss, "my_cls_acc" : my_cls_acc}

        return ret

    def save_std_pickle(self):
        '''
        class_std, sample_std = self.memory.get_std()
        self.class_std_list.append(class_std)
        self.sample_std_list.append(sample_std)
        '''
        
        with open('ours_cls_std_real.pickle', 'wb') as handle:
            pickle.dump(self.class_std_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open('ours_sample_std_real.pickle', 'wb') as handle:
            pickle.dump(self.sample_std_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    '''
    def report_training(self, sample_num, train_loss, train_acc):
        writer.add_scalar(f"train/loss", train_loss, sample_num)
        writer.add_scalar(f"train/acc", train_acc, sample_num)
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc, online_acc):
        writer.add_scalar(f"test/loss", avg_loss, sample_num)
        writer.add_scalar(f"test/acc", avg_acc, sample_num)
        writer.add_scalar(f"test/online_acc", online_acc, sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | online_acc {online_acc:.4f} "
        )
    '''

    def update_memory(self, sample, count):
        #self.reservoir_memory(sample)
        #self.memory.replace_sample(sample, count=count)
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            cand_idx = self.memory.cls_idx[cls_to_replace]
            idx_to_replace = random.choice(cand_idx) 
            #score = self.memory.others_loss_decrease[cand_idx]
            #idx_to_replace = cand_idx[np.argmin(score)]
            self.memory.replace_sample(sample, idx_to_replace)
        else:
            self.memory.replace_sample(sample)
