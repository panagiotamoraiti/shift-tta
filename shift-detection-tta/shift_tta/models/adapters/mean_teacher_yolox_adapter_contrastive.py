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
from mmcv.ops import soft_nms

from shift_tta.registry import MODELS
from .base_adapter import BaseAdapter


@MODELS.register_module()
class MeanTeacherYOLOXAdapterContrastive(BaseAdapter):
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
                 filter_pseudo_labels=None,
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
                 batch_data_samples: TrackDataSample, rescale: bool = True):
        """Detector forward pass."""
        feats = detector.extract_feat(img)
        outs = detector.bbox_head.forward(feats)

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        predictions = detector.bbox_head.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
            
        ### Define new config object with score_thr = 0
        cfg_orig = deepcopy(detector.bbox_head.test_cfg)
        cfg_orig['score_thr'] = 0
        orig_predictions = detector.bbox_head.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, with_nms=True, rescale=False, cfg=cfg_orig) ### Without filtering
            
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
               *args, **kwargs) -> InstanceData:
        """Adapt the model."""
        # Extract image width and height
        img_height = teacher_img.shape[2]
        img_width = teacher_img.shape[3]

        # teacher forward ### -> self.teacher.module is the teacher
        teacher_det_results, teacher_outs, teacher_feats, teacher_predictions, teacher_orig_predictions = self._detect_forward(
            self.teacher.module.detector, teacher_img, teacher_data_samples)
        teacher_outs = self._expand_view(teacher_outs, views=self.views)
              
        teacher_outs = dict(
            cls_score=teacher_outs[0],
            bbox_pred=teacher_outs[1],
            objectness=teacher_outs[2])

        # student forward ### -> model is the student
        _, outs, student_feats, student_predictions, student_orig_predictions = self._detect_forward(
            model.detector, student_imgs, student_data_samples)
        outs = dict(
            cls_score=outs[0],
            bbox_pred=outs[1],
            objectness=outs[2])    
            
  
        '''print(len(teacher_orig_predictions[0].bboxes))
        print(len(student_orig_predictions[0].bboxes))
        print()
        print(len(teacher_predictions[0].bboxes))
        print(len(student_predictions[0].bboxes))
        print()'''
            
        ### --Consistency-Contrastive loss (includes multi-scale features)
        loss, consistency_loss, contrastive_loss = self.loss(outs, teacher_outs, teacher_feats, student_feats, teacher_orig_predictions, student_orig_predictions, img_width, img_height)
        
        self.s +=1
        if self.s % self.optim_steps == 0:
            self.steps.append(self.s)
            self.final_losses.append(loss.detach().cpu().numpy())
            self.consistency_losses.append(consistency_loss.detach().cpu().numpy())
            self.contrastive_losses.append(contrastive_loss.detach().cpu().numpy())
            print("Step:", self.s//self.optim_steps)
            print("Consistency loss:", consistency_loss)
            print("Contrastive loss:", contrastive_loss)
            print("Final loss:", loss)
            print()
            
            ### -- Save augmented image of the student with bboxes
            '''teacher_img_vis = teacher_img[0].permute(1, 2, 0).cpu().numpy()
            teacher_img_vis = cv2.cvtColor(teacher_img_vis, cv2.COLOR_BGR2RGB)
            teacher_reg_boxes = teacher_orig_predictions[0].bboxes
            teacher_score = teacher_orig_predictions[0].scores
            teacher_label = teacher_orig_predictions[0].labels
            class_names = ["Pedestrian", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"]
            
            # Filter pseudo labels based on threshold
            if self.filter_pseudo_labels  is not None:
                mask = teacher_score >= self.filter_pseudo_labels 
                teacher_score = teacher_score[mask]
                teacher_label = teacher_label[mask]
                teacher_reg_boxes = teacher_reg_boxes[mask]
                
            for i in range(len(student_imgs)):
                student_img = student_imgs[i].permute(1, 2, 0).cpu().numpy()
                student_img = cv2.cvtColor(student_img, cv2.COLOR_BGR2RGB)   
                #print(student_img.shape[0]) # 608
                #print(student_img.shape[1]) # 960
   
                for box, label, score in zip(teacher_reg_boxes, teacher_label, teacher_score):
                    xmin, ymin, xmax, ymax = map(int, box)
                    cv2.rectangle(teacher_img_vis, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                    cv2.rectangle(student_img, (xmin-5, ymin-3), (xmax-5, ymax-3), (255, 0, 0), 1)  
        
                    # Add label and score
                    label_text = f"{class_names[label]} {score:.2f}"
                    (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(teacher_img_vis, (xmin, ymin - label_height - 8), (xmin + label_width, ymin - 2), (255, 255, 255), -1)
                    cv2.putText(teacher_img_vis, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

                # Remove normalization for visualization
                student_img = student_img.clip(0, 255).astype(np.uint8)
                teacher_img_vis = teacher_img_vis.clip(0, 255).astype(np.uint8)
                
                plt.subplot(1, 2, 1) 
                plt.imshow(teacher_img_vis)
                plt.title("Teacher Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(student_img)
                plt.title(f"Student Augmented View {i}")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'results/images/{self.s//5}_img_view{i}.png', dpi=500)
                plt.close() '''
            ### --

        epochs = 2400 # Number of images
        if self.s == epochs*self.optim_steps:
            print("Saving plots...")
            # Plot Consistency Loss
            plt.figure(figsize=(10, 6))
            plt.plot(self.steps, self.consistency_losses, label='Consistency Loss', linewidth=2, color='blue')
            plt.xlabel('Steps')
            plt.ylabel('Consistency Loss')
            plt.title('Consistency Loss over Steps')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('results/plots/consistency_loss_plot.png')
            plt.close()

            # Plot Contrastive Loss
            plt.figure(figsize=(10, 6))
            plt.plot(self.steps, self.contrastive_losses, label='Contrastive Loss', linewidth=2, color='green')
            plt.xlabel('Steps')
            plt.ylabel('Contrastive Loss')
            plt.title('Contrastive Loss over Steps')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('results/plots/contrastive_loss_plot.png')
            plt.close()

            # Plot Final Loss
            plt.figure(figsize=(10, 6))
            plt.plot(self.steps, self.final_losses, label='Final Loss', linewidth=2, color='red')
            plt.xlabel('Steps')
            plt.ylabel('Final Loss')
            plt.title('Final Loss over Steps')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('results/plots/final_loss_plot.png')
            plt.close()
        ### --

        # adapt student
        loss.backward()
        self.optim_wrapper.step()
        self.optim_wrapper.zero_grad()
        
        if self.fixed_source_model:
            # Fixed source model Forward
            _, _, _, fixed_model_predictions, _ = self._detect_forward(
                self.fixed_model.detector, teacher_img, teacher_data_samples)
                
            teacher_reg_boxes = teacher_predictions[0].bboxes
            teacher_score = teacher_predictions[0].scores
            teacher_labels = teacher_predictions[0].labels
            fixed_model_reg_boxes = fixed_model_predictions[0].bboxes
            fixed_model_score = fixed_model_predictions[0].scores
            fixed_model_labels = fixed_model_predictions[0].labels
            
            boxes = torch.cat((teacher_reg_boxes, fixed_model_reg_boxes), dim=0)
            scores = torch.cat((teacher_score, fixed_model_score), dim=0)
            labels = torch.cat((teacher_labels, fixed_model_labels), dim=0)
            
            # Apply Soft-NMS
            soft_nms_preds, index = soft_nms(
                boxes=boxes, # (N, 4)
                scores=scores, # (N,)
                iou_threshold=0.7,
                sigma=0.5,
                min_score=0.01,
                method='gaussian'
            )
            
            boxes = soft_nms_preds # (N, 4)
            scores = scores[index]
            labels = labels[index]
            
            print(soft_nms_preds)
            print(index)
            print(len(labels))
            
            print(fixed_model_predictions)
            
            # Convert detections to a list of InstanceData
            results_list_soft_nms = []
            for i in range(len(scores)):
                instance_data = InstanceData()
                instance_data.scores = scores[i].item()
                instance_data.labels = labels[i].item()
                instance_data.bboxes = soft_nms_preds[i].tolist()
                results_list_soft_nms.append(instance_data)
            
            # Detection results contain the detections after soft-nms from the teacher and the fixed source model
            det_results = self.fixed_model.detector.add_pred_to_datasample(
                teacher_data_samples, results_list_soft_nms)
            

        return teacher_det_results


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
                    teacher_data_samples, student_data_samples) ### Update Student
                self.teacher.update_parameters(model) ### Update Teacher
                
                ### --Stochastically restore student's weights
                if self.stochastic_restoration: 
                    for nm, m  in model.named_modules():
                        for npp, p in m.named_parameters():
                            if npp in ['weight', 'bias'] and p.requires_grad:
                                mask = (torch.rand(p.shape) < self.rst_prob).float().cuda() # For values < rst_thresh put 1, restore this weight
                                with torch.no_grad():
                                    p.data = self.source_model_state[f"{nm}.{npp}"] * mask + p * (1.0-mask) ### Restore weights using saved initial_model_state
                ### --

        self._reset_optimizer()

        # update pred_det_instances
        pred_det_instances = outs[0].pred_instances.clone()

        return pred_det_instances
