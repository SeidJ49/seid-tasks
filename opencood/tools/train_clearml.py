# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yue Hu <18671129361@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# NO Multi-GPU training
# Diffrent: scheduler.setp() instead of cosine annealing

import argparse
import os
import statistics
import sys
import logging

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

# ClearML integration
from clearml import Task

# Add the root path to the system path
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)


# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log', mode='w')
    ]
)

# Create logger for each module
logger = logging.getLogger(__name__)

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True, help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='', help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate", help='passed to inference.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()

    # Initialize ClearML task
    task = Task.init(project_name='bm2cp', task_name='adver_city_bm2cp_radar_camera')
    # Connect command-line arguments and hypes configuration to ClearML
    task.connect(vars(opt))

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    task.connect(hypes)

    logger.info('------------------------------------------------------------------------------------------------------------------------')
    logger.info('--------------------------------------------DATASET BUILDING------------------------------------------------------------')
    logger.info('------------------------------------------------------------------------------------------------------------------------')

    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=4
                              )
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=4,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=4
                            )

    logger.info('------------------------------------------------------------------------------------------------------------------------')
    logger.info('--------------------------------------------CREATING MODEL--------------------------------------------------------------')
    logger.info('------------------------------------------------------------------------------------------------------------------------')

    # Create the model
    model = train_utils.create_model(hypes)
    total_params = sum(param.nelement() for param in model.parameters())
    logger.info(f"Number of parameters: {total_params}")

    # Set the device to GPU if available and move the model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.to(device)

    # Define the loss function and setup the optimizer
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)

    # Load from checkpoint if specified, otherwise start from scratch
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch, n_iter_per_epoch=len(train_loader))
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes)
        logger.info(f"Output results saved to: {saved_path}")
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, n_iter_per_epoch=len(train_loader))

    # Initialize tracking variables
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # Setup TensorBoard writer
    writer = SummaryWriter(saved_path)

    logger.info('------------------------------------------------------------------------------------------------------------------------')
    logger.info('--------------------------------------------TRAINING START--------------------------------------------------------------')
    logger.info('------------------------------------------------------------------------------------------------------------------------')

    # Get the number of epochs from the hyperparameters
    epoches = hypes['train_params']['epoches']

    # Training loop
    for epoch in range(init_epoch, max(epoches, init_epoch)):

        # Log learning rate for each parameter group
        for param_group in optimizer.param_groups:
            logger.info(f'Learning rate: {param_group["lr"]:.6f}')

        # Iterate over the training dataset
        for i, batch_data in enumerate(train_loader):
            if batch_data is None:
                continue

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch

            output_dict = model(batch_data['ego'])
            final_loss = criterion(output_dict, batch_data['ego']['label_dict'])

            # Log training loss via ClearML
            global_step = epoch * len(train_loader) + i
            task.get_logger().report_scalar(title=r"Training Loss", series=r"total_loss", value=final_loss.item(), iteration=global_step)

            # Log the training loss via the criterion
            criterion.logging(epoch, i, len(train_loader), writer)

            # Save the training loss to a file
            with open(os.path.join(saved_path, 'train_loss.txt'), 'a+') as f:
                msg = f'Epoch[{epoch}], iter[{i}/{len(train_loader)}], loss[{final_loss}]. \n'
                f.write(msg)

            # Back-propagation
            final_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()


        # Save the model checkpoint periodically
        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(), os.path.join(saved_path, f'net_epoch{epoch + 1}.pth'))

        # Evaluate the model periodically
        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_losses = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    output_dict = model(batch_data['ego'])
                    val_loss = criterion(output_dict, batch_data['ego']['label_dict'])
                    valid_losses.append(val_loss.item())
                    torch.cuda.empty_cache()

            valid_ave_loss = statistics.mean(valid_losses)
            logger.info(f'At epoch {epoch}, the validation loss is {valid_ave_loss:.6f}')
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # Log validation loss via ClearML
            task.get_logger().report_scalar(title=r"Validation Loss", series=r"loss", value=valid_ave_loss, iteration=epoch)

            # Save the validation loss to a file
            validation_loss_msg = f'Epoch[{epoch}], loss[{valid_ave_loss}]'
            with open(os.path.join(saved_path, 'validation_loss.txt'), 'a+') as f:
                f.write(validation_loss_msg + "\n")

        # Step the validation loss to a file
        scheduler.step(epoch) # TODO: change to scheduler.step()

    logger.info('------------------------------------------------------------------------------------------------------------------------')
    logger.info('--------------------------------------------TRAINING FINISHED-----------------------------------------------------------')
    logger.info('------------------------------------------------------------------------------------------------------------------------')
    logger.info(f'Checkpoints saved to {saved_path}')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
