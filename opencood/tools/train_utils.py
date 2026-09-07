# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import glob
import importlib
import yaml
import os
import re
from datetime import datetime
import shutil
import torch
import torch.optim as optim
from functools import partial

from opencood.tools.optimization_fastai import OneCycle, OptimWrapper

def backup_script(full_path, folders_to_save=["models", "data_utils", "utils", "loss"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)


def load_saved_model_epoch(saved_path, model, epoch=None):
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    # if os.path.exists(os.path.join(saved_path, 'net_latest.pth')):
    #     model.load_state_dict(torch.load(os.path.join(saved_path, 'net_latest.pth')))
    # file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))

    if False:
        pass
    # if file_list:
    #    assert len(file_list) == 1
    #    model.load_state_dict(torch.load(file_list[0], map_location='cpu'), strict=False)
    #    return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model

    # if os.path.exists(os.path.join(saved_path, 'net_epoch_bestval*.pth')):
    #    model.load_state_dict(torch.load(os.path.join(saved_path, 'net_epoch_bestval*.pth')))
    #    return 100, model
    else:
        if epoch is None:
            initial_epoch = findLastCheckpoint(saved_path)
        else:
            initial_epoch = int(epoch)

        if initial_epoch > 0:
            print('resuming by loading epoch %d' % initial_epoch)

        state_dict_ = torch.load(os.path.join(saved_path, 'net_epoch%d.pth' % initial_epoch))
        state_dict = {}
        # convert data_parallal to model
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]

        model_state_dict = model.state_dict()

        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, ' \
                          'loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        return initial_epoch, model

def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
    if file_list:
        assert len(file_list) == 1
        print("resuming best validation model at epoch %d" % \
                eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")))
        model.load_state_dict(torch.load(file_list[0] , map_location='cpu'), strict=False)
        return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(
            os.path.join(saved_path,
                         'net_epoch%d.pth' % initial_epoch), map_location='cpu'), strict=False)

    return initial_epoch, model


def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
                backup_script(full_path)
            except FileExistsError:
                pass
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)



    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file.

    Supports the original torch optimizers and RadarDistill/OpenPCDet-style
    fastai Adam OneCycle via ``optimizer.core_method: adam_onecycle``.
    """
    method_dict = hypes['optimizer']
    core_method = method_dict['core_method']
    optimizer_args = method_dict.get('args', {})

    if core_method == 'adam_onecycle':
        betas = optimizer_args.get('betas', (0.9, 0.99))
        # YAML parses ``[0.9, 0.99]`` as a list, while Adam/FastAI-style
        # helpers expect beta pairs as tuples.
        if isinstance(betas, list):
            betas = tuple(betas)
        optimizer_func = partial(optim.Adam, betas=betas)
        return OptimWrapper.create(
            optimizer_func,
            method_dict['lr'],
            [model],
            wd=optimizer_args.get('weight_decay', 0.01),
            true_wd=optimizer_args.get('true_wd', True),
            bn_wd=optimizer_args.get('bn_wd', True),
        )

    optimizer_method = getattr(optim, core_method, None)
    if not optimizer_method:
        raise ValueError('{} is not supported'.format(core_method))
    if optimizer_args:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'],
                                **optimizer_args)
    return optimizer_method(model.parameters(), lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, init_epoch=None, steps_per_epoch=None):
    """
    Set up the learning-rate scheduler.

    OneCycle is stepped per batch. Legacy schedulers are stepped per epoch.
    """
    lr_schedule_config = hypes['lr_scheduler']
    last_epoch = init_epoch if init_epoch is not None else 0

    if lr_schedule_config['core_method'] in {'none', None}:
        class NoOpLRScheduler(object):
            step_per_batch = False

            def __init__(self, wrapped_optimizer):
                self.optimizer = wrapped_optimizer

            def step(self, *args, **kwargs):
                return None

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                return None

            def get_last_lr(self):
                return [group['lr'] for group in self.optimizer.param_groups]

        scheduler = NoOpLRScheduler(optimizer)

    elif lr_schedule_config['core_method'] == 'onecycle':
        if steps_per_epoch is None:
            steps_per_epoch = 1
        total_step = hypes['train_params']['epoches'] * max(int(steps_per_epoch), 1)
        scheduler = OneCycle(
            optimizer,
            total_step=total_step,
            lr_max=lr_schedule_config['max_lr'],
            moms=lr_schedule_config.get('moms', [0.95, 0.85]),
            div_factor=lr_schedule_config.get('div_factor', 10),
            pct_start=lr_schedule_config.get('pct_start', 0.4),
            last_step=last_epoch * max(int(steps_per_epoch), 1),
        )

    elif lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    else:
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    if not getattr(scheduler, 'step_per_batch', False):
        for _ in range(last_epoch):
            scheduler.step()

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str) or not hasattr(inputs, 'to'):
            return inputs
        return inputs.to(device, non_blocking=True)
