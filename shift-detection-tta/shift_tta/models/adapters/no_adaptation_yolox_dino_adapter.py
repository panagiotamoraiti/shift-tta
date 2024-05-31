# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from copy import deepcopy

from mmengine.dataset import Compose
from mmengine.optim import build_optim_wrapper
from mmengine.structures import InstanceData
from mmtrack.structures import TrackDataSample
from mmcv.ops import soft_nms, nms

from shift_tta.registry import MODELS
from .base_adapter import BaseAdapter

from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,)
from PIL import Image
from torchvision import transforms
from mmdet.models.utils.misc import samplelist_boxtype2tensor
import copy
from mmcv.ops.nms import batched_nms


@MODELS.register_module()
class NoAdaptationYOLOXDinoAdapter(BaseAdapter):
    """Mean-teacher YOLOX adapter model with contrastive loss.

    Args:
        teacher (dict): Configuration of teacher. Defaults to None.
        optim_wrapper (dict): Configuration of optimizer wrapper. 
            Defaults to None.
        loss (dict): Configuration of loss. Defaults to None.
        pipeline (list(dict)): Configuration of image transforms.
            Defaults to None.
    """

    def __init__(self,
                 teacher: Optional[dict] = None,
                 fixed_source_model: bool = False,
                 stochastic_restoration: bool = False,
                 rst_prob: float = 0.01,
                 optim_wrapper: Optional[dict] = None,
                 optim_steps: int = 0,
                 filter_pseudo_labels: int = 0,
                 loss: Optional[dict] = dict(
                     type='ROIConsistencyLoss',
                     weight_consistency_loss = 0.01,
                     weight_contrastive_loss = 0.01,
                     contrastive = False,
                     filter_pseudo_labels=None
                 ),
                 pipeline: Optional[list[dict]] = None,
                 teacher_pipeline: Optional[list[dict]] = None,
                 student_pipeline: Optional[list[dict]] = None,
                 views: int = 1,
                 plot: bool = True,
                 plot_augmented_imgs: bool = False,
                 dataset: str = 'shift',
                 **kwargs) -> None:
        super().__init__(**kwargs)  

        self.teacher = None
        if teacher is not None:
            self.teacher_cfg = teacher

        # build optimizer
        self.optim_wrapper = None
        if optim_wrapper is not None:
            self.optim_wrapper_cfg = optim_wrapper
        self.optim_steps = optim_steps

        # build loss
        self.loss = MODELS.build(loss)

        # build image transforms
        self.pipeline = Compose(pipeline)
        self.teacher_pipeline = Compose(teacher_pipeline)
        self.student_pipeline = Compose(student_pipeline)
        self.views = views
        self.stochastic_restoration = stochastic_restoration
        self.rst_prob = rst_prob
        self.filter_pseudo_labels = filter_pseudo_labels
        self.fixed_source_model = fixed_source_model
        
        # Initialize lists to track losses
        self.consistency_losses = []
        self.contrastive_losses = []
        self.final_losses = []
        self.steps = []
        self.s = 0
        self.plot = plot
        self.plot_augmented_imgs = plot_augmented_imgs
        self.dataset = dataset
        
        ### --Add Grounding Dino Inference
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model_dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)


        # TODO: implement param_scheduler for optimizer (e.g. lr decay)

    def _init_source_model_state(self, model) -> None:
        """Init self.source_model_state.
        
        Args:
            model (nn.Module): detection model.
        """
        super()._init_source_model_state(model)

        self.optim_wrapper = build_optim_wrapper(model, self.optim_wrapper_cfg)
        self.optim_wrapper_state = deepcopy(self.optim_wrapper.state_dict())

        if self.teacher_cfg is not None:
            self.teacher_cfg['model'] = model
            self.teacher = MODELS.build(self.teacher_cfg)
            self.teacher_model_state = deepcopy(self.teacher.state_dict())
            
        # Create a fixed source model
        if self.fixed_source_model:
            # Create a deep copy of the model for restoration
            self.fixed_model = deepcopy(model)
 

    def _reset_optimizer(self) -> None:
        """Reset optimizer state.
        
        Args:
            model (nn.Module): detection model."""
        if self.optim_wrapper is not None:
            self.optim_wrapper.load_state_dict(self.optim_wrapper_state)

    def _restore_source_model_state(self, model) -> None:
        """Init self.source_model_state.
        
        Args:
            model (nn.Module): detection model.
        """
        super()._restore_source_model_state(model)
            
        if self.teacher is not None:
            self.teacher.load_state_dict(
                self.teacher_model_state, strict=True)

    def _detect_forward(self, detector: torch.nn.Module, img: torch.Tensor, 
                 batch_data_samples: TrackDataSample, rescale: bool = True, fixed: bool = False):
        """Detector forward pass."""
        feats = detector.extract_feat(img)
        outs = detector.bbox_head.forward(feats)

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
                    
        predictions = detector.bbox_head.predict_by_feat(
                *outs, batch_img_metas=batch_img_metas, rescale=rescale)
            
        ### Define new config object with score_thr
        cfg_orig = deepcopy(detector.bbox_head.test_cfg)
        cfg_orig['score_thr'] = self.filter_pseudo_labels
        orig_predictions = detector.bbox_head.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, with_nms=True, rescale=False, cfg=cfg_orig) ### With other filtering threshold
            
        det_results = detector.add_pred_to_datasample(
            batch_data_samples, predictions)

        return det_results, outs, feats, predictions, orig_predictions

    def _expand_view(self, outs: Tuple[torch.Tensor], views: int = 1):
        """Expand batch size of each element in outs to views."""
        outs = tuple(list(o.repeat_interleave(views, dim=0) for o in out) 
                     for out in outs)
        return outs

    def _adapt(self, model: torch.nn.Module, 
               teacher_img: torch.Tensor, 
               student_imgs: torch.Tensor, 
               teacher_data_samples: List[TrackDataSample],
               student_data_samples: List[TrackDataSample],
               dino_data_samples: List[TrackDataSample],
               *args, **kwargs) -> InstanceData:
        # teacher forward ### -> self.teacher.module is the teacher
        teacher_det_results, teacher_outs, teacher_feats, teacher_predictions, teacher_orig_predictions = self._detect_forward(
            self.teacher.module.detector, teacher_img, teacher_data_samples)
           
            
        ### --Add Grounding Dino Inference
        if self.dataset == 'shift':
            text = "pedestrian. car. truck. bus. motorcycle. bicycle."
            class_names ={"pedestrian": 0, "car": 1, "truck": 2, "bus": 3, "motorcycle": 4, "bicycle": 5}
        elif self.dataset == 'cityscapes':
            text = "person. rider. car. train. motorcycle. bicycle. truck. bus."
            class_names ={"person": 0, "rider": 1, "car": 2, "train": 3, "motorcycle": 4, "bicycle": 5, "truck": 6, "bus": 7} 
        elif self.dataset == 'kitti':
            pass 
        elif self.dataset=="coco":
            text = 'person. bicycle. car. motorcycle. airplane. bus. train. truck. boat.'
            
            class_names = {
            "person": 0,
            "bicycle": 1,
            "car": 2,
            "motorcycle": 3,
            "airplane": 4,
            "bus": 5,
            "train": 6,
            "truck": 7,
            "boat": 8,
            "trafficlight": 9,
            "firehydrant": 10,
            "streetsign": 11,
            "stopsign": 12,
            "parkingmeter": 13,
            "bench": 14,
            "bird": 15,
            "cat": 16,
            "dog": 17,
            "horse": 18,
            "sheep": 19,
            "cow": 20,
            "elephant": 21,
            "bear": 22,
            "zebra": 23,
            "giraffe": 24,
            "hat": 25,
            "backpack": 26,
            "umbrella": 27,
            "shoe": 28,
            "eyeglasses": 29,
            "handbag": 30,
            "tie": 31,
            "suitcase": 32,
            "frisbee": 33,
            "skis": 34,
            "snowboard": 35,
            "sports ball": 36,
            "kite": 37,
            "baseballbat": 38,
            "baseballglove": 39,
            "skateboard": 40,
            "surfboard": 41,
            "tennisracket": 42,
            "bottle": 43,
            "plate": 44,
            "wineglass": 45,
            "cup": 46,
            "fork": 47,
            "knife": 48,
            "spoon": 49,
            "bowl": 50,
            "banana": 51,
            "apple": 52,
            "sandwich": 53,
            "orange": 54,
            "broccoli": 55,
            "carrot": 56,
            "hotdog": 57,
            "pizza": 58,
            "donut": 59,
            "cake": 60,
            "chair": 61,
            "couch": 62,
            "potted plant": 63,
            "bed": 64,
            "mirror": 65,
            "dining table": 66,
            "window": 67,
            "desk": 68,
            "toilet": 69,
            "door": 70,
            "tv": 71,
            "laptop": 72,
            "mouse": 73,
            "remote": 74,
            "keyboard": 75,
            "cell phone": 76,
            "microwave": 77,
            "oven": 78,
            "toaster": 79,
            "sink": 80,
            "refrigerator": 81,
            "blender": 82,
            "book": 83,
            "clock": 84,
            "vase": 85,
            "scissors": 86,
            "teddybear": 87,
            "hairdrier": 88,
            "toothbrush": 89,
            "hairbrush": 90
            }

        elif self.dataset == 'clad':
            pass

        to_pil = transforms.ToPILImage()
        
        # Convert tensor to PIL image
        image = to_pil(teacher_img[0, :, :, :])
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model_dino(**inputs)
        preds = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.5,
            text_threshold=0.5,
            target_sizes=[image.size[::-1]]
        )
        
        labels = preds[0]["labels"]
        # print(labels)
        scores = preds[0]["scores"]
        boxes = preds[0]["boxes"]
        
        labels = [label.split()[0] for label in labels]
        label_indexes = [class_names[label] for label in labels]
        label_indexes = torch.tensor(label_indexes)

        # Convert detections to a list of InstanceData
        results_list = []
        pred_det_instances = InstanceData()
        pred_det_instances.labels = label_indexes
        pred_det_instances.bboxes = boxes[:, :4] # (N, 4)
        pred_det_instances.scores = scores
        
        batch_img_metas = [
            data_samples.metainfo for data_samples in dino_data_samples
        ]
        for img_id, img_meta in enumerate(batch_img_metas):
            pred_det_instances = _bbox_post_process(pred_det_instances, cfg=self.teacher.module.detector.bbox_head.test_cfg, rescale=True, img_meta=img_meta, with_nms=True) 
        results_list.append(pred_det_instances)
        
        dino_det_results = add_pred_to_datasample(dino_data_samples, results_list)
        
        return dino_det_results


    def adapt(self, model: torch.nn.Module, img: torch.Tensor,
              feats: List[torch.Tensor], data_sample: TrackDataSample,
              **kwargs) -> InstanceData:
        """Adapt the model.
        
        Args:
            model (nn.Module): detection model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1 in
                ByteTrack method.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_det_instances`.

        Returns:
            :obj:`InstanceData`: Detection results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """

        metainfo = data_sample.metainfo
        frame_id = metainfo.get('frame_id', -1)
        if self.with_episodic and frame_id == 0: ### At the beginning of each video reset to source model
            self.reset(model)
            # print(model.detector)
        else:
            if self.source_model_state is None:
                self._init_source_model_state(model)

        # adapt model
        # TODO: apply multiple image transforms
        # data_sample = self.transforms(deepcopy(data_sample))
        # TODO: create a batch
        # TODO: compute teacher prediction on clean target image
        # TODO: compute distill loss into augmented batch 
        #   (concat targets to batch size)

        # make teacher and student views
        results = dict(img_path=data_sample.img_path,
                       instances=data_sample.instances,
        )
        results = self.pipeline(results)
        teacher_results = self.teacher_pipeline(results)
        teacher_img = teacher_results['inputs']['img'].to(img)
        teacher_data_samples = [teacher_results['data_samples']]
        
        dino_data_samples = copy.deepcopy(teacher_data_samples)
        

        student_imgs = []
        student_data_samples = []
        for _ in range(self.views):
            student_results = self.student_pipeline(results)
            student_imgs.append(student_results['inputs']['img'])
            student_data_samples.append(student_results['data_samples'])
        student_imgs = torch.cat(student_imgs).to(img)
        
        with torch.enable_grad():
            model.requires_grad_(True)
            model.train(True)
            for _ in range(self.optim_steps):
                ### Get teacher's detection results
                outs = self._adapt( 
                    model, teacher_img, student_imgs, 
                    teacher_data_samples, student_data_samples, dino_data_samples) ### Update Student
                self.teacher.update_parameters(model) ### Update Teacher

        self._reset_optimizer()

        # update pred_det_instances
        pred_det_instances = outs[0].pred_instances.clone()

        return pred_det_instances
        
        
def add_pred_to_datasample(data_samples, results_list):
    """Add predictions to `DetDataSample`.

    Args:
        data_samples (list[:obj:`DetDataSample`], optional): A batch of
            data samples that contain annotations and predictions.
        results_list (list[:obj:`InstanceData`]): Detection results of
            each image.

    Returns:
        list[:obj:`DetDataSample`]: Detection results of the
        input images. Each DetDataSample usually contain
        'pred_instances'. And the ``pred_instances`` usually
        contains following keys.

            - scores (Tensor): Classification scores, has a shape
                (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
                the last dimension 4 arrange as (x1, y1, x2, y2).
    """
    for data_sample, pred_instances in zip(data_samples, results_list):
        data_sample.pred_instances = pred_instances
    samplelist_boxtype2tensor(data_samples)
    return data_samples
    
def _bbox_post_process(results: InstanceData, cfg, rescale: bool = False, with_nms: bool = True, img_meta: Optional[dict] = None) -> InstanceData:
    """bbox post-processing method.

    The boxes would be rescaled to the original image scale and do
    the nms operation. Usually `with_nms` is False is used for aug test.

    Args:
        results (:obj:`InstaceData`): Detection instance results,
            each item has shape (num_bboxes, ).
        cfg (mmengine.Config): Test / postprocessing configuration,
            if None, test_cfg would be used.
        rescale (bool): If True, return boxes in original image space.
            Default to False.
        with_nms (bool): If True, do nms before return boxes.
            Default to True.
        img_meta (dict, optional): Image meta info. Defaults to None.

    Returns:
        :obj:`InstanceData`: Detection results of each image
        after the post process.
        Each item usually contains following keys.

        - scores (Tensor): Classification scores, has a shape
          (num_instance, )
        - labels (Tensor): Labels of bboxes, has a shape
          (num_instances, ).
        - bboxes (Tensor): Has a shape (num_instances, 4),
          the last dimension 4 arrange as (x1, y1, x2, y2).
    """

    if rescale:
        assert img_meta.get('scale_factor') is not None
        results.bboxes /= results.bboxes.new_tensor(
            img_meta['scale_factor']).repeat((1, 2))

    if with_nms and results.bboxes.numel() > 0:
        det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores,
                                            results.labels, cfg.nms)
        results = results[keep_idxs]
        # some nms would reweight the score, such as softnms
        results.scores = det_bboxes[:, -1]
    return results
