"""Fail-closed AP15 checkpoint/target contract validation."""

import os

from opencood.models.sub_modules.preservation import checkpoint_sha256


SELECTED_RADAR_ANCHOR = 'E20'
SELECTED_LIDAR_TEACHER = 'E8'
SELECTED_TARGET = 'P-Heatmap'
SELECTED_REGION = 'car_gaussian_3x3'
EXPECTED_RADAR_SHA256 = '9a49511a6df7e1e486796d6c4bc372485a9baef0df0176ae5c43d88508e1ac83'
EXPECTED_TEACHER_SHA256 = '918e58623e1d7c7376a3ae26c52378bea7750757e89353df3dd037ede958b14e'
EXPECTED_NORMALIZATION_SCALE = 0.11837511690693056


def _require(condition, message):
    if not condition:
        raise ValueError('[AP15 preservation contract] ' + message)


def _verified_checkpoint(role, path, expected_sha256, verify_hashes):
    _require(bool(path), '%s checkpoint path is missing.' % role)
    _require(os.path.isfile(path), '%s checkpoint does not exist: %s' % (role, path))
    actual = checkpoint_sha256(path) if verify_hashes else None
    if verify_hashes:
        _require(
            actual == expected_sha256,
            '%s SHA-256 mismatch: expected %s, got %s'
            % (role, expected_sha256, actual),
        )
    return {'path': path, 'sha256': actual or expected_sha256}


def validate_preservation_checkpoint_contract(hypes, verify_hashes=True):
    model_args = hypes.get('model', {}).get('args', {})
    config = model_args.get('wp3_preservation')
    if config is None:
        return {'active': False}

    enabled = bool(config.get('enabled', False))
    selections = {
        'radar_anchor': config.get('selected_radar_anchor'),
        'lidar_teacher': config.get('selected_lidar_teacher'),
        'target': config.get('target'),
        'region': config.get('region'),
    }
    expected = {
        'radar_anchor': SELECTED_RADAR_ANCHOR,
        'lidar_teacher': SELECTED_LIDAR_TEACHER,
        'target': SELECTED_TARGET,
        'region': SELECTED_REGION,
    }
    for key, value in expected.items():
        _require(selections[key] == value, '%s must be %s, got %s.' % (key, value, selections[key]))

    _require(bool(model_args.get('freeze_teacher', False)), 'LiDAR teacher must be frozen.')
    _require(
        abs(float(config.get('normalization_scale', 0.0)) - EXPECTED_NORMALIZATION_SCALE) < 1e-12,
        'normalization_scale must reproduce the AP15-0 E20 validation-anchor scale.',
    )
    _require(str(config.get('loss_type', '')).lower() == 'mse', 'AP15-1 binds L_pres to normalized MSE.')
    _require(int(config.get('region_radius', -1)) == 1, 'car_gaussian_3x3 requires region_radius=1.')
    _require(abs(float(config.get('gaussian_sigma', 0.0)) - 1.0) < 1e-12, 'gaussian_sigma must be 1.0.')
    lambda_pres = float(config.get('lambda_pres', 0.0))
    _require(lambda_pres > 0.0 if enabled else lambda_pres == 0.0,
             'lambda_pres must be positive when enabled and exactly zero when disabled.')

    student = _verified_checkpoint(
        'student/E20',
        hypes.get('train_params', {}).get('pretrained_model_path', ''),
        EXPECTED_RADAR_SHA256,
        verify_hashes,
    )
    teacher = _verified_checkpoint(
        'LiDAR teacher/E8',
        model_args.get('teacher_ckpt', ''),
        EXPECTED_TEACHER_SHA256,
        verify_hashes,
    )
    anchor = None
    if enabled:
        _require(config.get('anchor_sha256') == EXPECTED_RADAR_SHA256,
                 'anchor_sha256 must be the selected E20 digest.')
        anchor = _verified_checkpoint(
            'radar anchor/E20',
            config.get('anchor_checkpoint', ''),
            EXPECTED_RADAR_SHA256,
            verify_hashes,
        )

    return {
        'active': True,
        'enabled': enabled,
        'selection': selections,
        'student': student,
        'lidar_teacher': teacher,
        'radar_anchor': anchor,
        'lambda_pres': lambda_pres,
        'student_only_trainable_required': True,
    }


def format_contract_report(report):
    if not report.get('active', False):
        return '[AP15 preservation contract] inactive'
    mode = 'P1-v2 enabled' if report['enabled'] else 'C1-v2 disabled control'
    return (
        '[AP15 preservation contract] %s; anchor=%s; teacher=%s; target=%s; region=%s'
        % (
            mode,
            report['selection']['radar_anchor'],
            report['selection']['lidar_teacher'],
            report['selection']['target'],
            report['selection']['region'],
        )
    )
