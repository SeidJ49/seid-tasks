# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
import sys
from pathlib import Path
from typing import OrderedDict
import importlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.truckscenes_eval_utils import (
    build_prediction_record,
    run_official_truckscenes_eval,
)
from opencood.visualization import vis_utils, my_vis, simple_vis
torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='single',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--epoch', type=int, default=24, help='Epoch to load the model from')
    parser.add_argument('--benchmark', action='store_true',
                        help='Enable timing benchmark output')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations not counted in stats')
    parser.add_argument('--timing_mode', type=str, default='pipeline',
                        choices=['pipeline', 'e2e'],
                        help="pipeline: time just the model pipeline call; "
                             "e2e: time from after dataloader batch to end of inference")
    parser.add_argument('--max_bench_iters', type=int, default=-1,
                        help='Limit the number of timed iterations (-1 = all)')
    parser.add_argument('--official_eval', type=str, default='none',
                        choices=['none', 'truckscenes'],
                        help='run official dataset evaluation after OpenCOOD AP evaluation')
    parser.add_argument('--truckscenes_root', type=str, default=None,
                        help='TruckScenes dataroot; if omitted, infer from PKL sensor paths')
    parser.add_argument('--truckscenes_version', type=str, default='v1.1-mini',
                        help='TruckScenes version string for official devkit evaluation')
    parser.add_argument('--truckscenes_devkit_src', type=str, default='/home/nj644/dev/Studis/truckscenes-devkit/src',
                        help='Path to the TruckScenes devkit src directory')
    parser.add_argument('--eval_split', type=str, default='val', choices=['val', 'test'],
                        help='dataset split used for inference and evaluation')

    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    # if 'heter' in hypes:
    #     x_min, x_max = -140.8, 140.8
    #     y_min, y_max = -40, 40
    #     opt.note += f"_{x_max}_{y_max}"
    #     hypes['fusion']['args']['grid_conf']['xbound'] = [x_min, x_max, hypes['fusion']['args']['grid_conf']['xbound'][2]]
    #     hypes['fusion']['args']['grid_conf']['ybound'] = [y_min, y_max, hypes['fusion']['args']['grid_conf']['ybound'][2]]
    #     hypes['model']['args']['grid_conf'] = hypes['fusion']['args']['grid_conf']

    #     new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
    #                         x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]
        
    #     hypes['preprocess']['cav_lidar_range'] =  new_cav_range
    #     hypes['postprocess']['anchor_args']['cav_lidar_range'] = new_cav_range
    #     hypes['postprocess']['gt_range'] = new_cav_range
    #     hypes['model']['args']['lidar_args']['lidar_range'] = new_cav_range
    #     if 'camera_mask_args' in hypes['model']['args']:
    #         hypes['model']['args']['camera_mask_args']['cav_lidar_range'] = new_cav_range

    #     # reload anchor
    #     yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    #     for name, func in yaml_utils_lib.__dict__.items():
    #         if name == hypes["yaml_parser"]:
    #             parser_func = func
    #     hypes = parser_func(hypes)
        
    
    if opt.eval_split == 'test':
        hypes['validate_dir'] = hypes['test_dir']
        if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
            assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model_epoch(saved_path, model, epoch=opt.epoch)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = eval_utils.init_result_stat()
    class_names = hypes.get('model', {}).get('args', {}).get('class_names', [])
    class_result_stat = {class_name: eval_utils.init_result_stat() for class_name in class_names}

    
    infer_info = opt.fusion_method + opt.note
    inference_times_ms = []
    prediction_records = []

    for i, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        # Decide whether this iteration is timed (after warmup)
        do_time = opt.benchmark and (i >= opt.warmup)
        if opt.max_bench_iters > 0 and opt.benchmark and (i - opt.warmup) >= opt.max_bench_iters:
            break

        # E2E timing starts earlier (includes device transfer); pipeline starts after transfer
        if opt.benchmark and opt.timing_mode == 'e2e':
            if torch.cuda.is_available():
                e2e_start = torch.cuda.Event(enable_timing=True)
                e2e_end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                e2e_start.record()
            else:
                import time
                e2e_start_cpu = time.perf_counter()
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
             # Pipeline timing (right around the actual inference call)
            if opt.benchmark and opt.timing_mode == 'pipeline':
                if torch.cuda.is_available():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start.record()
                else:
                    import time
                    start_cpu = time.perf_counter()

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')
            # Stop pipeline timing
            if opt.benchmark and opt.timing_mode == 'pipeline':
                if torch.cuda.is_available():
                    end.record()
                    torch.cuda.synchronize()
                    elapsed_ms = start.elapsed_time(end)  # ms
                else:
                    elapsed_ms = (time.perf_counter() - start_cpu) * 1000.0
                if do_time:
                    inference_times_ms.append(elapsed_ms)

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            pred_label_tensor = infer_result.get('pred_label_tensor')
            gt_label_tensor = infer_result.get('gt_label_tensor')

            if opt.official_eval == 'truckscenes':
                prediction_records.append(
                    build_prediction_record(
                        batch_data=batch_data,
                        infer_result=infer_result,
                        class_names=class_names,
                        box_order=opencood_dataset.post_processor.params['order'],
                    )
                )
            
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            if class_result_stat:
                eval_utils.caluclate_tp_fp_multiclass(pred_box_tensor, pred_score, gt_box_tensor, pred_label_tensor, gt_label_tensor, class_result_stat, 0.3, class_names)
                eval_utils.caluclate_tp_fp_multiclass(pred_box_tensor, pred_score, gt_box_tensor, pred_label_tensor, gt_label_tensor, class_result_stat, 0.5, class_names)
                eval_utils.caluclate_tp_fp_multiclass(pred_box_tensor, pred_score, gt_box_tensor, pred_label_tensor, gt_label_tensor, class_result_stat, 0.7, class_names)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "lidar_agent_record": lidar_agent_record})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                """
                If you want 3D visualization, uncomment lines below
                """
                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,batch_data['ego']['origin_lidar'][0],hypes['postprocess']['gt_range'],vis_save_path,method='bev',left_hand=left_hand)
             # Stop e2e timing (after post-inference bits you want to include)
            if opt.benchmark and opt.timing_mode == 'e2e':
                if torch.cuda.is_available():
                    e2e_end.record()
                    torch.cuda.synchronize()
                    elapsed_ms = e2e_start.elapsed_time(e2e_end)
                else:
                    elapsed_ms = (time.perf_counter() - e2e_start_cpu) * 1000.0
                if do_time:
                    inference_times_ms.append(elapsed_ms)
        torch.cuda.empty_cache()
        if opt.benchmark and len(inference_times_ms) > 0:
            arr = np.array(inference_times_ms, dtype=np.float64)
            mean_ms = float(arr.mean())
            median_ms = float(np.median(arr))
            p90_ms = float(np.percentile(arr, 90))
            best_ms = float(arr.min())
            fps_mean = 1000.0 / mean_ms
            fps_best = 1000.0 / best_ms
            mode_str = 'Model pipeline' if opt.timing_mode == 'pipeline' else 'End-to-end'
            print(f"\n[Benchmark] {mode_str} (excluding warmup: {opt.warmup} iters)")
            print(f"  Samples timed: {len(arr)}")
            print(f"  Mean:   {mean_ms:.3f} ms  ({fps_mean:.2f} FPS)")
            print(f"  Median: {median_ms:.3f} ms")
            print(f"  P90:    {p90_ms:.3f} ms")
            print(f"  Best:   {best_ms:.3f} ms  ({fps_best:.2f} FPS)\n")

    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)
    if class_result_stat:
        eval_utils.eval_final_results_multiclass(class_result_stat, opt.model_dir, infer_info)

    if opt.official_eval == 'truckscenes':
        output_dir = os.path.join(opt.model_dir, f'official_truckscenes_{infer_info}')
        result_str, result_dict = run_official_truckscenes_eval(
            prediction_records=prediction_records,
            dataset=opencood_dataset,
            output_dir=output_dir,
            version=opt.truckscenes_version,
            dataroot=opt.truckscenes_root,
            devkit_src=opt.truckscenes_devkit_src,
        )
        print(result_str)
        print(f'Official TruckScenes metrics saved to {output_dir}')

if __name__ == '__main__':
    main()
