"""Fail-closed AP14-H selection contract for opportunity-guided PFD."""

from __future__ import absolute_import, division, print_function

import hashlib
from pathlib import Path


SELECTED_OPPORTUNITY_STUDENT = 'E20'
SELECTED_OPPORTUNITY_TEACHER = 'E8'
SELECTED_SIGNAL = 'O3_floor_need'
SELECTED_STUDENT_SHA256 = (
    '9a49511a6df7e1e486796d6c4bc372485a9baef0df0176ae5c43d88508e1ac83')
SELECTED_TEACHER_SHA256 = (
    '918e58623e1d7c7376a3ae26c52378bea7750757e89353df3dd037ede958b14e')


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _opportunity_config(hypes):
    model_args = hypes.get('model', {}).get('args', {})
    guided = model_args.get('wp3_guided_pfd', {}) or {}
    if str(guided.get('variant', '')) != 'opportunity':
        return None, model_args
    return guided.get('opportunity', {}) or {}, model_args


def validate_opportunity_checkpoint_contract(hypes, verify_files=True):
    """Reject any silent checkpoint/signal substitution for AP14-1.

    This check is inactive for native and legacy Guided-PFD variants.  For the
    opportunity variant it binds both enabled O1 and disabled C1 controls to
    the supervised AP14-H E20/E8 selection and, by default, hashes both files
    before any dataset or optimizer is created.
    """
    config, model_args = _opportunity_config(hypes)
    if config is None:
        return {'active': False}

    declared = {
        'selected_student': str(config.get('selected_student', '')),
        'selected_teacher': str(config.get('selected_teacher', '')),
        'signal': str(config.get('signal', '')),
        'student_checkpoint_sha256': str(config.get(
            'student_checkpoint_sha256', '')).lower(),
        'teacher_checkpoint_sha256': str(config.get(
            'teacher_checkpoint_sha256', '')).lower(),
    }
    expected = {
        'selected_student': SELECTED_OPPORTUNITY_STUDENT,
        'selected_teacher': SELECTED_OPPORTUNITY_TEACHER,
        'signal': SELECTED_SIGNAL,
        'student_checkpoint_sha256': SELECTED_STUDENT_SHA256,
        'teacher_checkpoint_sha256': SELECTED_TEACHER_SHA256,
    }
    mismatches = [
        '{}={!r}, expected {!r}'.format(key, declared[key], expected[key])
        for key in expected if declared[key] != expected[key]
    ]
    if mismatches:
        raise ValueError(
            'AP14-H opportunity selection contract mismatch: ' +
            '; '.join(mismatches))
    if bool(config.get('use_radar_evidence', False)):
        raise ValueError(
            'AP14-0 excludes radar evidence from the first opportunity variant.')
    teacher_floor = float(config.get('teacher_floor', -1.0))
    if abs(teacher_floor - 0.25) > 1e-12:
        raise ValueError('AP14-1 O3 requires teacher_floor=0.25.')

    student_path = Path(hypes.get('train_params', {}).get(
        'pretrained_model_path', ''))
    teacher_path = Path(model_args.get('teacher_ckpt', ''))
    result = {
        'active': True,
        'student': SELECTED_OPPORTUNITY_STUDENT,
        'teacher': SELECTED_OPPORTUNITY_TEACHER,
        'signal': SELECTED_SIGNAL,
        'student_checkpoint': str(student_path),
        'teacher_checkpoint': str(teacher_path),
        'student_checkpoint_sha256': SELECTED_STUDENT_SHA256,
        'teacher_checkpoint_sha256': SELECTED_TEACHER_SHA256,
    }
    if not verify_files:
        return result

    for role, path, expected_hash in (
            ('student', student_path, SELECTED_STUDENT_SHA256),
            ('teacher', teacher_path, SELECTED_TEACHER_SHA256)):
        if not path.is_file():
            raise FileNotFoundError(
                'AP14-1 selected {} checkpoint not found: {}'.format(role, path))
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                'AP14-1 {} checkpoint SHA-256 mismatch: {} != {}'.format(
                    role, actual_hash, expected_hash))
        result[role + '_checkpoint_sha256_actual'] = actual_hash
    return result


def format_contract_report(contract):
    if not contract.get('active', False):
        return '[Opportunity contract] inactive'
    return (
        '[Opportunity contract] student={student} teacher={teacher} '
        'signal={signal} student_sha256={student_checkpoint_sha256} '
        'teacher_sha256={teacher_checkpoint_sha256}'
    ).format(**contract)
