from opencood.data_utils.datasets.late_fusion_dataset_radar import SingleVehicleDatasetRadar

__all__ = {
    'SingleVehicleDatasetRadar': SingleVehicleDatasetRadar,
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
