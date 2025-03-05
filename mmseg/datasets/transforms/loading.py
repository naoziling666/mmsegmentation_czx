# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Union

import mmcv
import mmengine.fileio as fileio
import numpy as np
import cv2 as cv
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile

from mmseg.registry import TRANSFORMS
from mmseg.utils import datafrombytes
import os
try:
    from osgeo import gdal
except ImportError:
    gdal = None


@TRANSFORMS.register_module()
class LoadAnnotations(MMCV_LoadAnnotations):

    """Load annotations for semantic segmentation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str



@TRANSFORMS.register_module()
class LoadImageFromNpyFile_Train(BaseTransform):
    """Load an image from npy file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img_dir = os.listdir('/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/seafog_600/crop_image_3band/train')
        img_dir = img_dir[0:116]

        filename = results['img_path']
        img = np.load(filename)
        if not np.random.randint(4):
            if img.shape[2]==3:
                root = '/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/seafog_600/crop_image_3band/train'
                index_random = np.random.randint(len(img_dir))
                filename = os.path.join(root, img_dir[index_random])
                img = img = np.load(filename)
                png_name = img_dir[index_random].split('.')[0]+'.png'
                results['seg_map_path'] = os.path.join('/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/seafog_600/crop_mask/train', png_name)
            elif img.shape[2]==6:
                root = '/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/seafog_600/crop_image_6band/train'
                index_random = np.random.randint(len(img_dir))
                filename = os.path.join(root, img_dir[index_random])
                img = img = np.load(filename)
                png_name = img_dir[index_random].split('.')[0]+'.png'
                results['seg_map_path'] = os.path.join('/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/seafog_600/crop_mask/train', png_name)
            elif img.shape[2]==16:
                root = '/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/seafog_600/crop_image_16band/train'
                index_random = np.random.randint(len(img_dir))
                filename = os.path.join(root, img_dir[index_random])
                img = img = np.load(filename)
                png_name = img_dir[index_random].split('.')[0]+'.png'
                results['seg_map_path'] = os.path.join('/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/seafog_600/crop_mask/train', png_name)
        # if img.shape
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results
  

@TRANSFORMS.register_module()
class LoadImageFromNpyFile_Optical_flow_RAFT(BaseTransform):
    """Load an image from npy file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 optical_flow_dir = "",
                 preprocess_optical_flow = False,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.optical_flow_dir = optical_flow_dir
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.preprocess_optical_flow = preprocess_optical_flow
        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def calculate_optical_flow(self, image):

        def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
            """
            Expects a two dimensional flow image of shape.

            Args:
                flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
                clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
                convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

            Returns:
                np.ndarray: Flow visualization image of shape [H,W,3]
            """
            assert flow_uv.ndim == 3, 'input flow must have three dimensions'
            assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
            if clip_flow is not None:
                flow_uv = np.clip(flow_uv, 0, clip_flow)
            u = flow_uv[:,:,0]
            v = flow_uv[:,:,1]
            rad = np.sqrt(np.square(u) + np.square(v))
            rad_max = np.max(rad)
            epsilon = 1e-5
            u = u / (rad_max + epsilon)
            v = v / (rad_max + epsilon)
            return flow_uv_to_colors(u, v, convert_to_bgr)

        def make_colorwheel():
            """
            Generates a color wheel for optical flow visualization as presented in:
                Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
                URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

            Code follows the original C++ source code of Daniel Scharstein.
            Code follows the the Matlab source code of Deqing Sun.

            Returns:
                np.ndarray: Color wheel
            """

            RY = 15
            YG = 6
            GC = 4
            CB = 11
            BM = 13
            MR = 6

            ncols = RY + YG + GC + CB + BM + MR
            colorwheel = np.zeros((ncols, 3))
            col = 0

            # RY
            colorwheel[0:RY, 0] = 255
            colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
            col = col+RY
            # YG
            colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
            colorwheel[col:col+YG, 1] = 255
            col = col+YG
            # GC
            colorwheel[col:col+GC, 1] = 255
            colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
            col = col+GC
            # CB
            colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
            colorwheel[col:col+CB, 2] = 255
            col = col+CB
            # BM
            colorwheel[col:col+BM, 2] = 255
            colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
            col = col+BM
            # MR
            colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
            colorwheel[col:col+MR, 0] = 255
            return colorwheel

        def flow_uv_to_colors(u, v, convert_to_bgr=False):
            """
            Applies the flow color wheel to (possibly clipped) flow components u and v.

            According to the C++ source code of Daniel Scharstein
            According to the Matlab source code of Deqing Sun

            Args:
                u (np.ndarray): Input horizontal flow of shape [H,W]
                v (np.ndarray): Input vertical flow of shape [H,W]
                convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

            Returns:
                np.ndarray: Flow visualization image of shape [H,W,3]
            """
            flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
            colorwheel = make_colorwheel()  # shape [55x3]
            ncols = colorwheel.shape[0]
            rad = np.sqrt(np.square(u) + np.square(v))
            a = np.arctan2(-v, -u)/np.pi
            fk = (a+1) / 2*(ncols-1)
            k0 = np.floor(fk).astype(np.int32)
            k1 = k0 + 1
            k1[k1 == ncols] = 0
            f = fk - k0
            for i in range(colorwheel.shape[1]):
                tmp = colorwheel[:,i]
                col0 = tmp[k0] / 255.0
                col1 = tmp[k1] / 255.0
                col = (1-f)*col0 + f*col1
                idx = (rad <= 1)
                col[idx]  = 1 - rad[idx] * (1-col[idx])
                col[~idx] = col[~idx] * 0.75   # out of range
                # Note the 2-i => BGR instead of RGB
                ch_idx = 2-i if convert_to_bgr else i
                flow_image[:,:,ch_idx] = np.floor(255 * col)
            return flow_image



        def optical_flow(image1, image2):
            # calculate optical flow
            from .optical_flow_model import InputPadder
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
            image1 = image1.unsqueeze(0).to('cuda')
            image2 = image2.unsqueeze(0).to('cuda')
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image, image)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            return flow_up
        def vis_flow(flo):
            flo = flo[0].permute(1,2,0).cpu().numpy()
            flo = flow_to_image(flo)
            # flo[label==0] = [0,0,0]
            return flo
        import argparse
        from .optical_flow_model import RAFT
        import torch
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint", default='/root/autodl-fs/mmsegmentation_czx/mmseg/datasets/transforms/raft-things.pth')
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args()
        model = model = torch.nn.DataParallel(RAFT(args))
        image1 = image[...,:3]
        image2 = image[...,3:6]
        image3 = image[...,6:]
        flow12 = optical_flow(image1, image2)
        flow13 = optical_flow(image1, image3)
        flow23 = optical_flow(image2, image3)
        flow_mean = vis_flow((flow12+flow13+flow23)/3)



        return flow_mean

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        img = np.load(filename)
        name = filename.split('/')[-1].split('.')[0]+'.png'
        if os.path.exists(os.path.join(self.optical_flow_dir,name)):
            optical_flow = cv.imread(os.path.join(self.optical_flow_dir,name))
        else:
            optical_flow = self.calculate_optical_flow(img)
        assert optical_flow is not None, f'failed to load optical flow: {filename}'
        assert optical_flow.shape[:2] == img.shape[:2], f'optical flow shape {optical_flow.shape[:2]} does not match image shape {img.shape[:2]}'
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)
        optical_flow = optical_flow.astype(img.dtype)
        # if self.preprocess_optical_flow:
            
        #     mean = np.array([174.37, 236.88, 228.24])
        #     std = np.array([44.88, 15.54, 20.47])
        #     optical_flow = (optical_flow - mean) / std


        img = np.concatenate((img, optical_flow), axis=2)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results





    



@TRANSFORMS.register_module()
class LoadImageFromNpyFile_Optical_flow(BaseTransform):
    """Load an image from npy file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def calculate_optical_flow(self, image):

        def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
            """
            Expects a two dimensional flow image of shape.

            Args:
                flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
                clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
                convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

            Returns:
                np.ndarray: Flow visualization image of shape [H,W,3]
            """
            assert flow_uv.ndim == 3, 'input flow must have three dimensions'
            assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
            if clip_flow is not None:
                flow_uv = np.clip(flow_uv, 0, clip_flow)
            u = flow_uv[:,:,0]
            v = flow_uv[:,:,1]
            rad = np.sqrt(np.square(u) + np.square(v))
            rad_max = np.max(rad)
            epsilon = 1e-5
            u = u / (rad_max + epsilon)
            v = v / (rad_max + epsilon)
            return flow_uv_to_colors(u, v, convert_to_bgr)

        def make_colorwheel():
            """
            Generates a color wheel for optical flow visualization as presented in:
                Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
                URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

            Code follows the original C++ source code of Daniel Scharstein.
            Code follows the the Matlab source code of Deqing Sun.

            Returns:
                np.ndarray: Color wheel
            """

            RY = 15
            YG = 6
            GC = 4
            CB = 11
            BM = 13
            MR = 6

            ncols = RY + YG + GC + CB + BM + MR
            colorwheel = np.zeros((ncols, 3))
            col = 0

            # RY
            colorwheel[0:RY, 0] = 255
            colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
            col = col+RY
            # YG
            colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
            colorwheel[col:col+YG, 1] = 255
            col = col+YG
            # GC
            colorwheel[col:col+GC, 1] = 255
            colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
            col = col+GC
            # CB
            colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
            colorwheel[col:col+CB, 2] = 255
            col = col+CB
            # BM
            colorwheel[col:col+BM, 2] = 255
            colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
            col = col+BM
            # MR
            colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
            colorwheel[col:col+MR, 0] = 255
            return colorwheel

        def flow_uv_to_colors(u, v, convert_to_bgr=False):
            """
            Applies the flow color wheel to (possibly clipped) flow components u and v.

            According to the C++ source code of Daniel Scharstein
            According to the Matlab source code of Deqing Sun

            Args:
                u (np.ndarray): Input horizontal flow of shape [H,W]
                v (np.ndarray): Input vertical flow of shape [H,W]
                convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

            Returns:
                np.ndarray: Flow visualization image of shape [H,W,3]
            """
            flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
            colorwheel = make_colorwheel()  # shape [55x3]
            ncols = colorwheel.shape[0]
            rad = np.sqrt(np.square(u) + np.square(v))
            a = np.arctan2(-v, -u)/np.pi
            fk = (a+1) / 2*(ncols-1)
            k0 = np.floor(fk).astype(np.int32)
            k1 = k0 + 1
            k1[k1 == ncols] = 0
            f = fk - k0
            for i in range(colorwheel.shape[1]):
                tmp = colorwheel[:,i]
                col0 = tmp[k0] / 255.0
                col1 = tmp[k1] / 255.0
                col = (1-f)*col0 + f*col1
                idx = (rad <= 1)
                col[idx]  = 1 - rad[idx] * (1-col[idx])
                col[~idx] = col[~idx] * 0.75   # out of range
                # Note the 2-i => BGR instead of RGB
                ch_idx = 2-i if convert_to_bgr else i
                flow_image[:,:,ch_idx] = np.floor(255 * col)
            return flow_image



        def optical_flow(image1, image2):
            # calculate optical flow
            img_1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
            img_2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(img_1, img_2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # 将光流的笛卡尔坐标转换为极坐标
            # magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

            # # 创建HSV图像，色相代表方向，亮度代表大小
            # hsv = np.zeros((img_1.shape[0], img_1.shape[1],3))
            # hsv[..., 1] = 255
            # hsv[..., 0] = angle * 180 / np.pi / 2
            # hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

            # 将HSV图像转换为BGR格式
            # bgr = cv.cvtColor(hsv.astype(np.float32), cv.COLOR_HSV2BGR)
            bgr = flow_to_image(flow)
            return bgr

        image1 = image[...,:3]
        image2 = image[...,3:6]
        image3 = image[...,6:]
        flow12 = optical_flow(image1, image2)
        flow13 = optical_flow(image1, image3)
        flow23 = optical_flow(image2, image3)
        flow = np.mean([flow12, flow13, flow23], axis=0)


        return flow

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        img = np.load(filename)
        optical_flow = self.calculate_optical_flow(img)

        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)
        optical_flow = optical_flow.astype(img.dtype)
        img = np.concatenate((img, optical_flow), axis=2)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results




@TRANSFORMS.register_module()
class LoadImageFromNpyFile(BaseTransform):
    """Load an image from npy file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        # try:
        #     if self.file_client_args is not None:
        #         file_client = fileio.FileClient.infer_client(
        #             self.file_client_args, filename)
        #         img_bytes = file_client.get(filename)
        #     else:
        #         img_bytes = fileio.get(
        #             filename, backend_args=self.backend_args)
        #     img = mmcv.imfrombytes(
        #         img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        # except Exception as e:
        #     if self.ignore_empty:
        #         return None
        #     else:
        #         raise e
        img = np.load(filename)
        # img = img[:,:,0:3] # for 3band of multiband
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

@TRANSFORMS.register_module()
class LoadImageFromNDArray(LoadImageFromFile):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadBiomedicalImageFromFile(BaseTransform):
    """Load an biomedical mage from file.

    Required Keys:

    - img_path

    Added Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities, and data type is float32
        if set to_float32 = True, or float64 if decode_backend is 'nifti' and
        to_float32 is False.
    - img_shape
    - ori_shape

    Args:
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an float64 array.
            Defaults to True.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 decode_backend: str = 'nifti',
                 to_xyz: bool = False,
                 to_float32: bool = True,
                 backend_args: Optional[dict] = None) -> None:
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']

        data_bytes = fileio.get(filename, self.backend_args)
        img = datafrombytes(data_bytes, backend=self.decode_backend)

        if self.to_float32:
            img = img.astype(np.float32)

        if len(img.shape) == 3:
            img = img[None, ...]

        if self.decode_backend == 'nifti':
            img = img.transpose(0, 3, 2, 1)

        if self.to_xyz:
            img = img.transpose(0, 3, 2, 1)

        results['img'] = img
        results['img_shape'] = img.shape[1:]
        results['ori_shape'] = img.shape[1:]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class LoadBiomedicalAnnotation(BaseTransform):
    """Load ``seg_map`` annotation provided by biomedical dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'gt_seg_map': np.ndarray (X, Y, Z) or (Z, Y, X)
        }

    Required Keys:

    - seg_map_path

    Added Keys:

    - gt_seg_map (np.ndarray): Biomedical seg map with shape (Z, Y, X) by
        default, and data type is float32 if set to_float32 = True, or
        float64 if decode_backend is 'nifti' and to_float32 is False.

    Args:
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        to_float32 (bool): Whether to convert the loaded seg map to a float32
            numpy array. If set to False, the loaded image is an float64 array.
            Defaults to True.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See :class:`mmengine.fileio` for details.
            Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 decode_backend: str = 'nifti',
                 to_xyz: bool = False,
                 to_float32: bool = True,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__()
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        data_bytes = fileio.get(results['seg_map_path'], self.backend_args)
        gt_seg_map = datafrombytes(data_bytes, backend=self.decode_backend)

        if self.to_float32:
            gt_seg_map = gt_seg_map.astype(np.float32)

        if self.decode_backend == 'nifti':
            gt_seg_map = gt_seg_map.transpose(2, 1, 0)

        if self.to_xyz:
            gt_seg_map = gt_seg_map.transpose(2, 1, 0)

        results['gt_seg_map'] = gt_seg_map
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class LoadBiomedicalData(BaseTransform):
    """Load an biomedical image and annotation from file.

    The loading data format is as the following:

    .. code-block:: python

        {
            'img': np.ndarray data[:-1, X, Y, Z]
            'seg_map': np.ndarray data[-1, X, Y, Z]
        }


    Required Keys:

    - img_path

    Added Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.
    - img_shape
    - ori_shape

    Args:
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 with_seg=False,
                 decode_backend: str = 'numpy',
                 to_xyz: bool = False,
                 backend_args: Optional[dict] = None) -> None:  # noqa
        self.with_seg = with_seg
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        data_bytes = fileio.get(results['img_path'], self.backend_args)
        data = datafrombytes(data_bytes, backend=self.decode_backend)
        # img is 4D data (N, X, Y, Z), N is the number of protocol
        img = data[:-1, :]

        if self.decode_backend == 'nifti':
            img = img.transpose(0, 3, 2, 1)

        if self.to_xyz:
            img = img.transpose(0, 3, 2, 1)

        results['img'] = img
        results['img_shape'] = img.shape[1:]
        results['ori_shape'] = img.shape[1:]

        if self.with_seg:
            gt_seg_map = data[-1, :]
            if self.decode_backend == 'nifti':
                gt_seg_map = gt_seg_map.transpose(2, 1, 0)

            if self.to_xyz:
                gt_seg_map = gt_seg_map.transpose(2, 1, 0)
            results['gt_seg_map'] = gt_seg_map
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'with_seg={self.with_seg}, '
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class InferencerLoader(BaseTransform):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(
            dict(type='LoadImageFromFile', **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='LoadImageFromNDArray', **kwargs))

    def transform(self, single_input: Union[str, np.ndarray, dict]) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if isinstance(single_input, str):
            inputs = dict(img_path=single_input)
        elif isinstance(single_input, np.ndarray):
            inputs = dict(img=single_input)
        elif isinstance(single_input, dict):
            inputs = single_input
        else:
            raise NotImplementedError

        if 'img' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)


@TRANSFORMS.register_module()
class LoadSingleRSImageFromFile(BaseTransform):
    """Load a Remote Sensing mage from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is a float64 array.
            Defaults to True.
    """

    def __init__(self, to_float32: bool = True):
        self.to_float32 = to_float32

        if gdal is None:
            raise RuntimeError('gdal is not installed')

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        ds = gdal.Open(filename)
        if ds is None:
            raise Exception(f'Unable to open file: {filename}')
        img = np.einsum('ijk->jki', ds.ReadAsArray())

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str


@TRANSFORMS.register_module()
class LoadMultipleRSImageFromFile(BaseTransform):
    """Load two Remote Sensing mage from file.

    Required Keys:

    - img_path
    - img_path2

    Modified Keys:

    - img
    - img2
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is a float64 array.
            Defaults to True.
    """

    def __init__(self, to_float32: bool = True):
        if gdal is None:
            raise RuntimeError('gdal is not installed')
        self.to_float32 = to_float32

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        filename2 = results['img_path2']

        ds = gdal.Open(filename)
        ds2 = gdal.Open(filename2)

        if ds is None:
            raise Exception(f'Unable to open file: {filename}')
        if ds2 is None:
            raise Exception(f'Unable to open file: {filename2}')

        img = np.einsum('ijk->jki', ds.ReadAsArray())
        img2 = np.einsum('ijk->jki', ds2.ReadAsArray())

        if self.to_float32:
            img = img.astype(np.float32)
            img2 = img2.astype(np.float32)

        if img.shape != img2.shape:
            raise Exception(f'Image shapes do not match:'
                            f' {img.shape} vs {img2.shape}')

        results['img'] = img
        results['img2'] = img2
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str
