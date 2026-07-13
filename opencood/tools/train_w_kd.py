# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
import argparse
import os
import statistics

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import importlib
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
import glob
from icecream import ic


def _format_state_key_report(keys, max_keys=20):
    keys = list(keys)
    if not keys:
        return "none"
    preview = keys[:max_keys]
    suffix = "" if len(keys) <= max_keys else f" ... (+{len(keys) - max_keys} more)"
    return ", ".join(preview) + suffix


def _load_student_checkpoint(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    state_dict = {
        (k[7:] if isinstance(k, str) and k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }
    model_state_dict = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state_dict and value.shape == model_state_dict[key].shape:
            compatible_state[key] = value
        else:
            skipped.append(key)
    missing = [key for key in model_state_dict if key not in compatible_state]
    model_state_dict.update(compatible_state)
    model.load_state_dict(model_state_dict, strict=False)
    print(f"[Fine-tune] initialized student from: {checkpoint_path}")
    print(f"[Fine-tune] loaded tensors: {len(compatible_state)}")
    print(f"[Fine-tune] skipped keys ({len(skipped)}): {_format_state_key_report(skipped)}")
    print(f"[Fine-tune] missing keys ({len(missing)}): {_format_state_key_report(missing)}")


def _load_teacher_state_dict(teacher_model, checkpoint_path, strict=False):
    teacher_state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(teacher_state, dict):
        if 'state_dict' in teacher_state:
            teacher_state = teacher_state['state_dict']
        elif 'model_state_dict' in teacher_state:
            teacher_state = teacher_state['model_state_dict']
    teacher_state = {
        (k[7:] if isinstance(k, str) and k.startswith('module.') else k): v
        for k, v in teacher_state.items()
    }
    load_result = teacher_model.load_state_dict(teacher_state, strict=strict)
    missing = getattr(load_result, 'missing_keys', [])
    unexpected = getattr(load_result, 'unexpected_keys', [])
    print(f"[KD teacher] checkpoint: {checkpoint_path}")
    print(f"[KD teacher] strict load: {strict}")
    print(f"[KD teacher] loaded tensors: {len(teacher_state)}")
    print(f"[KD teacher] missing keys ({len(missing)}): {_format_state_key_report(missing)}")
    print(f"[KD teacher] unexpected keys ({len(unexpected)}): {_format_state_key_report(unexpected)}")
    return missing, unexpected


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=8,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=8,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # define the loss
    criterion = train_utils.create_loss(hypes)

    train_params = hypes.get('train_params', {})
    if not opt.model_dir:
        pretrained_model_path = train_params.get('pretrained_model_path', '')
        pretrained_model_dir = train_params.get('pretrained_model_dir', '')
        pretrained_model_epoch = train_params.get('pretrained_model_epoch', None)
        fine_tune = bool(train_params.get('fine_tune', False) or pretrained_model_path or pretrained_model_dir)
        if pretrained_model_path:
            _load_student_checkpoint(model, pretrained_model_path)
        elif pretrained_model_dir:
            _, model = train_utils.load_saved_model_epoch(
                pretrained_model_dir, model, epoch=pretrained_model_epoch
            )
            print(
                f"[Fine-tune] initialized student from dir: {pretrained_model_dir} "
                f"epoch={pretrained_model_epoch if pretrained_model_epoch is not None else 'last'}"
            )
        if fine_tune:
            print(
                f"[Fine-tune] new run will train initialized weights for "
                f"{train_params.get('epoches')} fine-tuning epochs. Do not pass --model_dir for this mode."
            )

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(
            hypes, optimizer, init_epoch=init_epoch, steps_per_epoch=max(len(train_loader), 1))

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(
            hypes, optimizer, steps_per_epoch=max(len(train_loader), 1))

    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single

    ############ For DiscoNet ##############
    if "kd_flag" in hypes.keys():
        kd_flag = True
        teacher_model_name = hypes['kd_flag']['teacher_model'] # point_pillar_disconet_teacher
        teacher_model_config = hypes['kd_flag']['teacher_model_config']
        teacher_checkpoint_path = hypes['kd_flag']['teacher_path']

        # import the model
        model_filename = "opencood.models." + teacher_model_name
        model_lib = importlib.import_module(model_filename)
        teacher_model_class = None
        target_model_name = teacher_model_name.replace('_', '')

        for name, cls in model_lib.__dict__.items():
            if name.lower() == target_model_name.lower():
                teacher_model_class = cls

        teacher_model = teacher_model_class(teacher_model_config)
        teacher_strict = bool(hypes['kd_flag'].get('strict_teacher_load', False))
        _load_teacher_state_dict(teacher_model, teacher_checkpoint_path, strict=teacher_strict)

        for p in teacher_model.parameters():
            p.requires_grad_(False)

        if torch.cuda.is_available():
            teacher_model.to(device)

        teacher_model.eval()
    else:
        kd_flag = False
    ########################################

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        for i, batch_data in enumerate(train_loader):
            if batch_data is None:
                continue
            # the model will be evaluation mode during validation
            model.train()
            if getattr(scheduler, 'step_per_batch', False):
                scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)

            batch_data['ego']['epoch'] = epoch
            ouput_dict = model(batch_data['ego'])

            if kd_flag:
                teacher_output_dict = teacher_model(batch_data['ego'])
                ouput_dict.update(teacher_output_dict)

            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            # back-propagation
            final_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    if hypes.get('loss', {}).get('core_method') in {
                        'radardistill_loss',
                        'pillarnet_feature_kd_loss',
                        'pillarnet_feature_kd_scale_norm_loss',
                        'pillarnet_feedback_kd_loss',
                    }:
                        # CenterHead/PillarNet normally skips target assignment
                        # in eval mode. Validation still needs output_dict['loss'].
                        batch_data['ego']['compute_loss'] = True
                    ouput_dict = model(batch_data['ego'])

                    if kd_flag:
                        teacher_output_dict = teacher_model(batch_data['ego'])
                        ouput_dict.update(teacher_output_dict)

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        if not getattr(scheduler, 'step_per_batch', False):
            scheduler.step(epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)
    torch.cuda.empty_cache()
    run_test = True

    # ddp training may leave multiple bestval
    bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))

    if len(bestval_model_list) > 1:
        import numpy as np
        bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
        ascending_idx = np.argsort(bestval_model_epoch_list)
        for idx in ascending_idx:
            if idx != (len(bestval_model_list) - 1):
                os.remove(bestval_model_list[idx])

    if run_test:
        fusion_method = opt.fusion_method

        if 'noise_setting' in hypes and hypes['noise_setting']['add_noise']:
            cmd = f"python opencood/tools/inference_w_noise.py --model_dir {saved_path} --fusion_method {fusion_method}"
        else:
            cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
