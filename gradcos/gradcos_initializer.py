import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import get_dist_info
from torch import distributed as dist
from mmcv.parallel import scatter_kwargs

from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import get_root_logger

from gradcos.gradinit_optimizers import RescaleAdam


class SubsetLoader:
    def __init__(self, loader, num_iter=1):
        self.loader = loader
        self.num_iter = num_iter

    def __iter__(self):
        self.loader.sampler.generator.manual_seed(0)
        iterator = iter(self.loader)
        for i in range(self.num_iter):
            yield next(iterator)


def get_ordered_params(net):
    param_list = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.LayerNorm):
            param_list.append(m.weight)
            if m.bias is not None:
                param_list.append(m.bias)

    return param_list


def set_bn_modes(net):
    """Switch the BN layers into training mode, but does not track running stats.
    """
    for n, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = True
            m.track_running_stats = False


def recover_bn_modes(net):
    for n, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True


class GradCosInitializer:
    def __init__(self,
                 dataset_cfg,
                 workers_per_gpu,
                 samples_per_gpu,
                 gradinit_min_scale=0.01,
                 gradinit_lr=0.05,
                 gradinit_grad_clip=1.,
                 overlap=0.,
                 gradinit_gamma=5,
                 num_iters=100,
                 num_batch_splits=2,
                 opt_target="gc",
                 gn_mul=1.,
                 resnet=False,
                 sampler_cfg=None,
                 train_meta_file=None,
                 tasks=('val1', 'train1', 'val2'),
                 eval_interval=10,
                 fix_bias=False,
                 ):
        if train_meta_file is not None:
            dataset_cfg.ann_file = train_meta_file
        self.dataset = build_dataset(dataset_cfg)
        _samples_per_gpu = samples_per_gpu
        _workers_per_gpu = workers_per_gpu
        loader_cfg = dict(
            samples_per_gpu=_samples_per_gpu,
            workers_per_gpu=_workers_per_gpu,
            dist=dist.is_initialized(),
            round_up=True,
            seed=0,
            sampler_cfg=sampler_cfg,
            persistent_workers=_workers_per_gpu > 0,
            generator=torch.Generator()
        )
        self.data_loader = build_dataloader(self.dataset, **loader_cfg)
        self.val_loader = build_dataloader(self.dataset, **loader_cfg)

        self.gradinit_min_scale = gradinit_min_scale
        self.gradinit_lr = gradinit_lr
        self.gradinit_grad_clip = gradinit_grad_clip
        self.overlap = overlap
        self.gradinit_gamma = gradinit_gamma
        self.num_iters = num_iters
        self.resnet = resnet

        self.num_batch_splits = num_batch_splits
        self.opt_target = opt_target
        assert opt_target in ["gc", "gc_gn", "gv", "gv_gn", "gv_gn_sqrt", "gc_gn_sqrt"]
        self.gn_mul = gn_mul
        self.tasks = tasks
        self.eval_interval = eval_interval

        self.logger = get_root_logger()

        self.fix_bias = fix_bias

    def train(self, model, optimizer, params_list, task_name):
        self.logger.info("----Running the {} mode".format(task_name))
        self.data_loader.sampler.generator.manual_seed(0)
        train_iters = 0
        for idx, sample in enumerate(self.data_loader):
            train_iters += 1
            if torch.cuda.is_available():
                _, sample = scatter_kwargs(None, sample, target_gpus=[torch.cuda.current_device()])
                sample = sample[0]
            gt_label = sample.pop('gt_label')
            outputs = model(return_loss=False, softmax=False, post_process=False, **sample)
            gt_label = gt_label.to(device=outputs.device)
            loss = model.head.loss(outputs, gt_label, reduction_override="none")['loss']

            bs = gt_label.shape[0]
            all_grads, all_gnorm = batch_settings(
                bs_size=bs,
                chunks=self.num_batch_splits,
                r=self.overlap,
                init_loss=loss,
                params_list=params_list,
                is_train=True
            )

            gnorm_mean = all_gnorm.mean()

            sim = cal_cos_similarity(all_grads, all_gnorm)
            var = cal_grad_variance(all_grads)
            total_var = var.sum()

            flag = False
            if gnorm_mean > self.gradinit_gamma:
                flag = True

            self.logger.info(
                "GradCosine init no.{:04d}, gradnorm: {}, gradcos : {}, gradvar : {}, is_constraint : {}".format(
                    idx, gnorm_mean.item(), sim.item(), total_var.item(), flag))
            if flag:
                gnorm = gnorm_mean
                optimizer.zero_grad()
                gnorm.backward()
                optimizer.step(is_constraint=True)
            else:
                # obj_loss = - sim
                if self.opt_target == "gv_gn":
                    obj_loss = total_var - all_gnorm.square().sum() * self.gn_mul
                elif self.opt_target == "gv":
                    obj_loss = total_var
                elif self.opt_target == 'gc_gn':
                    obj_loss = - sim - all_gnorm.square().sum() * self.gn_mul
                elif self.opt_target == 'gc':
                    obj_loss = - sim
                elif self.opt_target == 'gv_gn_sqrt':
                    obj_loss = total_var - all_gnorm.sum() * self.gn_mul
                elif self.opt_target == 'gc_gn_sqrt':
                    obj_loss = - sim - all_gnorm.sum() * self.gn_mul
                optimizer.zero_grad()
                obj_loss.backward()
                optimizer.step(is_constraint=False)
            model.zero_grad()
            yield sim.item(), total_var.item()

            if train_iters >= self.num_iters:
                break
        return

    def val(self, model, params_list, task_name):
        self.logger.info("----Running the {} mode".format(task_name))
        self.val_loader.sampler.generator.manual_seed(0)
        val_iters = 0
        for idx, sample in enumerate(self.val_loader):
            val_iters += 1
            if torch.cuda.is_available():
                _, sample = scatter_kwargs(None, sample, target_gpus=[torch.cuda.current_device()])
                sample = sample[0]
            gt_label = sample.pop('gt_label')
            outputs = model(return_loss=False, softmax=False, post_process=False, **sample)
            gt_label = gt_label.to(device=outputs.device)
            loss = model.head.loss(outputs, gt_label, reduction_override="none")['loss']

            bs = gt_label.shape[0]
            all_grads, all_gnorm = batch_settings(
                bs_size=bs,
                chunks=self.num_batch_splits,
                r=self.overlap,
                init_loss=loss,
                params_list=params_list,
                is_train=False
            )

            gnorm_mean = all_gnorm.mean()

            sim = cal_cos_similarity(all_grads, all_gnorm)
            var = cal_grad_variance(all_grads)
            total_var = var.sum()

            flag = False
            if gnorm_mean > self.gradinit_gamma:
                flag = True

            self.logger.info(
                "GradCosine init no.{:04d}, gradnorm: {}, gradcos : {}, gradvar : {}, is_constraint : {}".format(
                    idx, gnorm_mean.item(), sim.item(), total_var.item(), flag))
            model.zero_grad()
            yield sim.item(), total_var.item()

            if val_iters >= self.num_iters:
                break

        return

    def run(self, model):
        self.logger.info("Initializing Network By GradCosine")
        if self.fix_bias:
            bias_params = [p for n, p in model.named_parameters() if n.endswith('.bias')]
            weight_params = [p for n, p in model.named_parameters() if n.endswith('.weight')]
            self.logger.info("Fix bias")
        else:
            bias_params = [p for n, p in model.named_parameters() if 'bias' in n]
            weight_params = [p for n, p in model.named_parameters() if 'weight' in n]

        optimizer = RescaleAdam(
            [{'params': weight_params, 'min_scale': self.gradinit_min_scale, 'lr': self.gradinit_lr},
             {'params': bias_params, 'min_scale': 0, 'lr': self.gradinit_lr}],
            grad_clip=self.gradinit_grad_clip)

        model.eval()
        if self.resnet:
            set_bn_modes(model)
        params_list = get_ordered_params(model)

        _, world_size = get_dist_info()

        tasks = self.tasks
        results = {}
        results_2 = {}
        train_it = iter(self.train(model, optimizer, params_list, 'train'))

        for task in tasks:
            self.logger.info("----Running the {} mode".format(task))
            gradcos_all = []
            gradvar_all = []
            if task.startswith('train'):
                for idx in range(self.eval_interval):
                    try:
                        gradcos, gradvar = next(train_it)
                    except StopIteration:
                        train_it = iter(self.train(model, optimizer, params_list, 'train'))
                        try:
                            gradcos, gradvar = next(train_it)
                        except StopIteration:
                            raise ValueError("The iterator cannot be with 0 len")
                    gradcos_all.append(gradcos)
                    gradvar_all.append(gradvar)
            else:
                for idx, (gradcos, gradvar) in enumerate(self.val(model, params_list, task)):
                    gradcos_all.append(gradcos)
                    gradvar_all.append(gradvar)

            results[task] = np.array(gradcos_all).mean().item()
            results_2[task + 'var'] = np.array(gradvar_all).mean().item()

        self.logger.info("The results : {}".format(results))
        self.logger.info("The extra results : {}".format(results_2))
        if self.resnet:
            recover_bn_modes(model)
        del optimizer
        del self.data_loader
        del self.dataset
        model.train()
        model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def batch_settings(bs_size, chunks, r, init_loss, params_list, is_train):
    length = int(bs_size / chunks)
    all_grads = []
    all_grads_flat = []
    all_gnorm = []
    for i in range(chunks):
        if i == 0:
            all_grads.append(
                torch.autograd.grad(init_loss[0:int((1 + r) * length * 1)].mean(),
                                    params_list, retain_graph=True, create_graph=is_train))
        elif i == chunks - 1:
            all_grads.append(
                torch.autograd.grad(init_loss[int((1 - r) * length * i):].mean(),
                                    params_list, retain_graph=True, create_graph=is_train))
        else:
            all_grads.append(
                torch.autograd.grad(init_loss[int((1 - r) * length * i):int((1 + r) * length * (i + 1))].mean(),
                                    params_list, retain_graph=True, create_graph=is_train))
    for grads in all_grads:
        grad_temp = torch.cat([torch.flatten(g) for g in grads]).view(1, -1)
        all_grads_flat.append(grad_temp)
        all_gnorm.append(grad_temp.square().sum().sqrt())

    all_grads_flat = torch.cat(all_grads_flat, 0)
    all_gnorm = torch.stack(all_gnorm)
    return all_grads_flat, all_gnorm


def cal_cos_similarity(grads, norm):
    D = grads.shape[0]
    similarity = torch.mm(grads, grads.t())
    norm2 = torch.mm(norm[:, None], norm[None])
    similarity = similarity / norm2
    similarity = similarity - torch.diag(torch.diag(similarity))
    return sum(sum(similarity)) / (D * D - D)


def cal_grad_variance(grads):
    return torch.var(grads, dim=0, unbiased=False)
