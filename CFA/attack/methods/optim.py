import numpy as np
import torch
from .base import BaseAttacker
from torch.optim import Optimizer
import torch.nn.functional as F


class OptimAttacker(BaseAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)

    # @property
    # def param_groups(self):
    #     return self.optimizer.param_groups

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def patch_update(self, **kwargs):
        self.optimizer.step()
        # grad = self.optimizer.param_groups[0]['params'][0].grad
        # print(torch.mean(torch.abs(grad)))
        self.patch_obj.clamp_(p_min=self.min_epsilon, p_max=self.max_epsilon)

    def attack_loss(self, confs):
        self.optimizer.zero_grad()
        loss = self.loss_fn(confs=confs, patch=self.detector_attacker.universal_patch[0])
        tv_loss, obj_loss = loss.values()
        tv_loss = torch.max(self.cfg.tv_eta * tv_loss, torch.tensor(0.1).to(self.device))
        loss = tv_loss.to(obj_loss.device) + obj_loss
        out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss}
        return out


class CFAOptimAttacker(BaseAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)

    # @property
    # def param_groups(self):
    #     return self.optimizer.param_groups

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def patch_update(self, **kwargs):
        self.optimizer.step()
        # grad = self.optimizer.param_groups[0]['params'][0].grad
        # print(torch.mean(torch.abs(grad)))
        self.patch_obj.clamp_(p_min=self.min_epsilon, p_max=self.max_epsilon)


    def non_targeted_attack(self, ori_tensor_batch, detector, **kwargs):
        losses = []
        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()

            ori_tensor_batch2 = ori_tensor_batch.clone().requires_grad_(True)
            self.detector_attacker.clear_fm_grad()
            detector.zero_grad()
            ori_tensor_batch2 = ori_tensor_batch2.to(detector.device)
            bboxes, confs, cls_array = detector(ori_tensor_batch2).values()

            # for yolov5
            selected_confs = confs.max(dim=-1, keepdim=True)[0]
            temp_loss = selected_confs.mean()
            temp_loss.backward(retain_graph=True)

            mid_feature = self.detector_attacker.mid_feature_map
            mid_back_grad = self.detector_attacker.mid_back_grad
            deep_feature = self.detector_attacker.deep_feature_map
            deep_back_grad = self.detector_attacker.deep_back_grad

            weighted_mid_feature = mid_feature * mid_back_grad
            weighted_deep_feature = deep_feature * deep_back_grad

            resize_weighted_deep_feature = weighted_deep_feature.view(weighted_deep_feature.shape[0], 128, 4, 13,
                                                                      13).mean(dim=2)
            weighted_deep_feature = F.interpolate(resize_weighted_deep_feature, size=(52, 52), mode='bilinear',
                                                  align_corners=False)

            p_mask = (weighted_deep_feature > 0) & (weighted_mid_feature > 0)
            p_feature = weighted_mid_feature * weighted_deep_feature * p_mask.float()
            # n_mask = (weighted_deep_feature < 0) & (weighted_mid_feature < 0)
            # n_feature = weighted_mid_feature * weighted_deep_feature * n_mask.float()
            # cfa_loss = p_feature.mean() + n_feature.mean()
            cfa_loss = p_feature.mean()

            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            adv_tensor_batch = adv_tensor_batch.to(detector.device)
            bboxes, confs, cls_array = detector(adv_tensor_batch).values()

            if hasattr(self.cfg, 'class_specify'):
                # TODO: only support filtering a single class now
                attack_cls = int(self.cfg.ATTACK_CLASS)
                confs = torch.cat(
                    ([conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
            elif hasattr(self.cfg, 'topx_conf'):
                # attack top x confidence
                # print(confs.size())
                confs = torch.sort(confs, dim=-1, descending=True)[0][:, :self.cfg.topx_conf]
                confs = torch.mean(confs, dim=-1)
            else:
                # only attack the max confidence
                confs = confs.max(dim=-1, keepdim=True)[0]

            detector.zero_grad()
            loss_dict = self.attack_loss(confs=confs, extr_loss=cfa_loss)
            loss = loss_dict['loss']
            loss.backward()
            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update()
        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

    def attack_loss(self, confs, **kwargs):
        extr_loss = kwargs['extr_loss']
        self.optimizer.zero_grad()
        loss = self.loss_fn(confs=confs, patch=self.detector_attacker.universal_patch[0], extr_loss=extr_loss)
        tv_loss, obj_loss, extr_loss = loss.values()
        tv_loss = torch.max(self.cfg.tv_eta * tv_loss, torch.tensor(0.1).to(self.device))
        loss = tv_loss.to(obj_loss.device) + obj_loss + extr_loss
        out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss, 'extr_loss': extr_loss}

        return out
