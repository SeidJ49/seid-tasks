from opencood.data_utils.datasets.lidar.single_dataset_lidar_baseline import SingleDatasetLidarBaseline

from opencood.data_utils.datasets.lidar_radar.single_dataset_lidar_radar_baseline import SingleDatasetLidarRadarBaseline
from opencood.data_utils.datasets.lidar_radar.single_dataset_lidar_radar_baseline_attention import SingleDatasetLidarRadarBaselineAttention
from opencood.data_utils.datasets.lidar_radar.single_dataset_lidar_radar_baseline_attention_mlp import SingleDatasetLidarRadarBaselineAttentionMlp
from opencood.data_utils.datasets.old.single_dataset_lidar_radar_baseline_attention_mlp_his import SingleDatasetLidarRadarBaselineAttentionMlpHis
from opencood.data_utils.datasets.lidar_radar.single_dataset_lidar_radar_baseline_attention_mlp_his_sweep import SingleDatasetLidarRadarBaselineAttentionMlpHisSweep

from opencood.data_utils.datasets.old.single_dataset_lidar_radar_baseline_his import SingleDatasetLidarRadarBaselineHis


__all__ = {
    # --- LiDAR --------------------------------------------------------------------------------------------------------
    'SingleDatasetLidarBaseline': SingleDatasetLidarBaseline,
    # ------------------------------------------------------------------------------------------------------------------

    # --- LiDAR_RADAR --------------------------------------------------------------------------------------------------
    'SingleDatasetLidarRadarBaseline': SingleDatasetLidarRadarBaseline,
    'SingleDatasetLidarRadarBaselineAttention': SingleDatasetLidarRadarBaselineAttention,
    'SingleDatasetLidarRadarBaselineAttentionMlp': SingleDatasetLidarRadarBaselineAttentionMlp,
    'SingleDatasetLidarRadarBaselineAttentionMlpHis': SingleDatasetLidarRadarBaselineAttentionMlpHis,
    'SingleDatasetLidarRadarBaselineAttentionMlpHisSweep': SingleDatasetLidarRadarBaselineAttentionMlpHisSweep,

    'SingleDatasetLidarRadarBaselineHis': SingleDatasetLidarRadarBaselineHis,
    # ------------------------------------------------------------------------------------------------------------------
}

def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
