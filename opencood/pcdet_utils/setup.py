import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PREFIX = 'opencood.pcdet_utils.'


def make_cuda_ext(name, module, sources):
    module_path = module
    if module_path.startswith(PACKAGE_PREFIX):
        module_path = module_path[len(PACKAGE_PREFIX):]

    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(THIS_DIR, *module_path.split('.'), src) for src in sources]
    )
    return cuda_ext


setup(
    name='pcd utils',
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[make_cuda_ext(
                name='iou3d_nms_cuda',
                module='opencood.pcdet_utils.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu'
                ]),
        make_cuda_ext(
            name='roiaware_pool3d_cuda',
            module='opencood.pcdet_utils.roiaware_pool3d',
            sources=[
                'src/roiaware_pool3d.cpp',
                'src/roiaware_pool3d_kernel.cu',
            ]
        ),
        make_cuda_ext(
            name='pointnet2_stack_cuda',
            module='opencood.pcdet_utils.pointnet2.pointnet2_stack',
            sources=[
                'src/pointnet2_api.cpp',
                'src/ball_query.cpp',
                'src/ball_query_gpu.cu',
                'src/group_points.cpp',
                'src/group_points_gpu.cu',
                'src/sampling.cpp',
                'src/sampling_gpu.cu',
                'src/interpolate.cpp',
                'src/interpolate_gpu.cu',
                'src/voxel_query_gpu.cu',
                'src/voxel_query.cpp'
            ],
        ),
        make_cuda_ext(
            name='pointnet2_batch_cuda',
            module='opencood.pcdet_utils.pointnet2.pointnet2_batch',
            sources=[
                'src/pointnet2_api.cpp',
                'src/ball_query.cpp',
                'src/ball_query_gpu.cu',
                'src/group_points.cpp',
                'src/group_points_gpu.cu',
                'src/interpolate.cpp',
                'src/interpolate_gpu.cu',
                'src/sampling.cpp',
                'src/sampling_gpu.cu',
            ],
        ),
        make_cuda_ext(
            name='bev_pool_ext',
            module='opencood.pcdet_utils.bev_pool',
            sources=[
                'src/bev_pool.cpp',
                'src/bev_pool_cuda.cu',
            ],
        ),
        make_cuda_ext(
            name='ingroup_inds_cuda',
            module='opencood.pcdet_utils.ingroup_inds',
            sources=[
                'src/ingroup_inds.cpp',
                'src/ingroup_inds_kernel.cu',
            ],
        ),
        make_cuda_ext(
            name='roipoint_pool3d_cuda',
            module='opencood.pcdet_utils.roipoint_pool3d',
            sources=[
                'src/roipoint_pool3d.cpp',
                'src/roipoint_pool3d_kernel.cu',
            ],
        )]

)