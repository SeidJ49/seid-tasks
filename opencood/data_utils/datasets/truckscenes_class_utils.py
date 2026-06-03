import numpy as np


TRUCKSCENES_NAME_TO_DETECTION = {
    'vehicle.car': 'car',
    'car': 'car',
    'vehicle.truck': 'truck',
    'truck': 'truck',
    'vehicle.bus.rigid': 'bus',
    'vehicle.bus.bendy': 'bus',
    'bus': 'bus',
    'vehicle.trailer': 'trailer',
    'vehicle.ego_trailer': 'trailer',
    'trailer': 'trailer',
    'vehicle.construction': 'other_vehicle',
    'vehicle.other': 'other_vehicle',
    'other_vehicle': 'other_vehicle',
}


def build_class_id_lookup(gt_names, object_ids, class_names):
    lookup = {}
    for gt_name, object_id in zip(gt_names, object_ids):
        canonical_name = TRUCKSCENES_NAME_TO_DETECTION.get(str(gt_name))
        if canonical_name not in class_names:
            continue
        lookup[int(object_id)] = class_names.index(canonical_name) + 1
    return lookup


def attach_class_ids(object_bbx_center, object_bbx_mask, object_ids, class_id_lookup):
    object_bbx_center = np.asarray(object_bbx_center)
    object_bbx_mask = np.asarray(object_bbx_mask)

    object_bbx_center_with_cls = np.zeros((object_bbx_center.shape[0], object_bbx_center.shape[1] + 1), dtype=object_bbx_center.dtype)
    filtered_mask = np.zeros_like(object_bbx_mask)
    filtered_ids = []

    next_index = 0
    for source_index, object_id in enumerate(object_ids):
        if source_index >= object_bbx_center.shape[0] or object_bbx_mask[source_index] == 0:
            continue

        class_id = class_id_lookup.get(int(object_id), 0)
        if class_id <= 0:
            continue

        object_bbx_center_with_cls[next_index, :-1] = object_bbx_center[source_index]
        object_bbx_center_with_cls[next_index, -1] = class_id
        filtered_mask[next_index] = 1
        filtered_ids.append(object_id)
        next_index += 1

    return object_bbx_center_with_cls, filtered_mask, filtered_ids