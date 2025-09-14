from opencood.data_utils.datasets.single_dataset_baseline import SingleDatasetBaseline
from opencood.data_utils.datasets.single_dataset_baseline_attention import SingleDatasetBaselineAttention
from opencood.data_utils.datasets.single_dataset_baseline_attention_mlp import SingleDatasetBaselineAttentionMlp
from opencood.data_utils.datasets.single_dataset_baseline_attention_mlp_his import SingleDatasetBaselineAttentionMlpHis

from opencood.data_utils.datasets.single_dataset_baseline_his import SingleDatasetBaselineHis

__all__ = {
    'SingleDatasetBaseline': SingleDatasetBaseline,
    'SingleDatasetBaselineAttention': SingleDatasetBaselineAttention,
    'SingleDatasetBaselineAttentionMlp': SingleDatasetBaselineAttentionMlp,
    'SingleDatasetBaselineAttentionMlpHis': SingleDatasetBaselineAttentionMlpHis,

    'SingleDatasetBaselineHis': SingleDatasetBaselineHis,
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
