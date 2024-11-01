import torch
import torch.nn as nn
import torch.nn.functional as F

from src.helpers.masked_predictions import MaskedPredictionModule

def involvement_label_smoothing(label, involvement, alpha=0.2):
    if label == 1:
        if involvement < 0.4:
            inv_ls = (1-alpha/2)*(involvement) + alpha/20
        elif involvement < 0.65:
            inv_ls = (1-alpha)*(involvement) + alpha/10
        else:
            inv_ls = involvement
        return inv_ls
    else:
        return 0

def involvement_tolerant_loss(patch_logits, patch_labels, core_indices, involvement):
    batch_size = len(involvement)
    loss = torch.tensor(0, dtype=torch.float32, device=patch_logits.device)
    for i in range(batch_size):
        patch_logits_for_core = patch_logits[core_indices == i]
        patch_labels_for_core = patch_labels[core_indices == i]
        involvement_for_core = involvement[i]
        if patch_labels_for_core[0].item() == 0:
            # core is benign, so label noise is assumed to be low
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_logits_for_core, patch_labels_for_core
            )
        elif involvement_for_core.item() > 0.65:
            # core is high involvement, so label noise is assumed to be low
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_logits_for_core, patch_labels_for_core
            )
        else:
            # core is of intermediate involvement, so label noise is assumed to be high.
            # we should be tolerant of the model's "false positives" in this case.
            pred_index_sorted_by_cancer_score = torch.argsort(
                patch_logits_for_core[:, 0], descending=True
            )
            patch_logits_for_core = patch_logits_for_core[
                pred_index_sorted_by_cancer_score
            ]
            patch_labels_for_core = patch_labels_for_core[
                pred_index_sorted_by_cancer_score
            ]
            n_predictions = patch_logits_for_core.shape[0]
            patch_predictions_for_core_for_loss = patch_logits_for_core[
                : int(n_predictions * involvement_for_core.item())
            ]
            patch_labels_for_core_for_loss = patch_labels_for_core[
                : int(n_predictions * involvement_for_core.item())
            ]
            loss += nn.functional.binary_cross_entropy_with_logits(
                patch_predictions_for_core_for_loss,
                patch_labels_for_core_for_loss,
            )


def simple_mil_loss(
    patch_logits,
    patch_labels,
    core_indices,
    top_percentile=0.2,
    pos_weight=torch.tensor(1.0),
):
    ce_loss = nn.functional.binary_cross_entropy_with_logits(
        patch_logits, patch_labels, pos_weight=pos_weight, reduction="none"
    )

    loss = torch.tensor(0, dtype=torch.float32, device=patch_logits.device)

    for i in torch.unique(core_indices):
        patch_losses_for_core = ce_loss[core_indices == i]
        n_patches = len(patch_losses_for_core)
        n_patches_to_keep = int(n_patches * top_percentile)
        patch_losses_for_core_sorted = torch.sort(patch_losses_for_core)[0]
        patch_losses_for_core_to_keep = patch_losses_for_core_sorted[:n_patches_to_keep]
        loss += patch_losses_for_core_to_keep.mean()

    return loss



class CancerDetectionLossBase(nn.Module):
    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement):
        raise NotImplementedError

class ConsistencyLoss(CancerDetectionLossBase):
    def __init__(
        self,
        consistency_mode: str = "distinct",
        loss_pos_weight: float = 1.0,
        prostate_mask: bool = True,
        needle_mask: bool = True,
        ood_mask: bool = True,
        weak_factor: float = 0.5,
        strong_factor: float = 0.5,
    ):
        super().__init__()
        self.consistency_mode = consistency_mode
        self.loss_pos_weight = loss_pos_weight
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask
        self.ood_mask = ood_mask
        self.weak_factor = weak_factor
        self.strong_factor = strong_factor

    def forward(self, logits_w, logits_s, prostate_mask, needle_mask, ood_mask, label, involvement):
        masks = []
        for i in range(len(logits_w)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            if self.prostate_mask:
                mask &= prostate_mask[i] > 0.5
            if self.needle_mask:
                mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)

        preds_w, B_w = MaskedPredictionModule()(logits_w, masks)
        preds_s, B_s = MaskedPredictionModule()(logits_s, masks)
        loss = torch.tensor(0, dtype=torch.float32, device=preds_w.device)
        preds_ood, _ = MaskedPredictionModule()(logits_w, ood_mask > 0.5)
        labels = torch.zeros(len(preds_w), device=preds_w.device)
        for i in range(len(preds_w)):
            labels[i] = label[B_w[i]]
        labels = labels[..., None]  # needs to match N, C shape of preds
        labels_ood = torch.zeros(len(preds_ood), device=preds_ood.device)
        preds_ood = preds_ood.squeeze()
        confidence_w = torch.maximum(preds_w, 1 - preds_w)
        pseudo_mask = preds_w * (confidence_w > 0.6).float()

        if self.consistency_mode == "distinct":
            L_w = nn.functional.binary_cross_entropy_with_logits(
                preds_w,
                labels,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=preds_w.device
                ),
            )

            L_s = nn.functional.binary_cross_entropy_with_logits(
                preds_s,
                pseudo_mask,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=preds_s.device
                ),
            )

            loss += self.weak_factor * L_w + self.strong_factor * L_s

        elif self.consistency_mode == "distinct_obp":
            L_s = nn.functional.binary_cross_entropy_with_logits(
                preds_w,
                labels,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=preds_w.device
                ),
            )

            L_u = nn.functional.binary_cross_entropy_with_logits(
                preds_w,
                preds_s,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=preds_s.device
                ),
            )

            obp = nn.functional.binary_cross_entropy_with_logits(
                preds_ood, 
                labels_ood, 
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=preds_w.device
                ),
            )

            loss += self.loss_pos_weight * self.weak_factor * L_s + self.strong_factor * L_u + (labels_ood.shape[0] / labels.shape[0]) * obp

        elif self.consistency_mode == "inv_aware": # involvement-aware MSE
            if str(involvement.item()).lower() == "nan":
                involvement = 0.0

            preds = self.weak_factor * preds_w + self.strong_factor * preds_s
            preds = preds / (self.weak_factor + self.strong_factor)
            
            loss_unreduced = nn.functional.mse_loss(
                preds, (labels * involvement).float(), reduction="none"
            ).float()
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()
            loss = loss.float()
        
        elif self.consistency_mode == "mixed":
            preds_ws = self.weak_factor * preds_w + self.strong_factor * preds_s
            preds_avg = preds_ws / (self.weak_factor + self.strong_factor)

            L_s = nn.functional.binary_cross_entropy_with_logits(
                preds_w,
                labels,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=preds_w.device
                ),
            )

            L_u = nn.functional.binary_cross_entropy_with_logits(
                preds_avg,
                pseudo_mask,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=preds_avg.device
                ),
            )

            loss += self.loss_pos_weight * self.weak_factor * L_s + self.strong_factor * L_u

        elif self.consistency_mode == "avg":
            preds = self.weak_factor * preds_w + self.strong_factor * preds_s
            preds = preds / (self.weak_factor + self.strong_factor)
            loss += nn.functional.binary_cross_entropy_with_logits(
                preds,
                labels,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=preds.device
                ),
            )        

        elif self.consistency_mode == "avg_obp":
            preds = self.weak_factor * preds_w + self.strong_factor * preds_s
            preds = preds / (self.weak_factor + self.strong_factor)
            ce = nn.functional.binary_cross_entropy_with_logits(
                preds,
                labels,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=preds.device
                ),
            )

            obp = nn.functional.binary_cross_entropy_with_logits(
                preds_ood, 
                labels_ood, 
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=preds_w.device
                ),
            )

            loss += ce + (labels_ood.shape[0] / labels.shape[0]) * obp

        elif self.consistency_mode == "avg_inv_obp":
            preds = self.weak_factor * preds_w + self.strong_factor * preds_s
            preds = preds / (self.weak_factor + self.strong_factor)
            ce += nn.functional.binary_cross_entropy_with_logits(
                preds,
                involvement,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=preds.device
                ),
            )

            obp = nn.functional.binary_cross_entropy_with_logits(
                preds_ood, 
                labels_ood, 
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=preds_w.device
                ),
            )

            loss += ce + (labels_ood.shape[0] / labels.shape[0]) * obp

        return loss

        


class CancerDetectionValidRegionLoss(CancerDetectionLossBase):
    def __init__(
        self,
        base_loss: str = "ce",
        loss_pos_weight: float = 1.0,
        prostate_mask: bool = True,
        needle_mask: bool = True,
        inv_label_smoothing: bool = False,
        smoothing_factor: float = 0.2,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.loss_pos_weight = loss_pos_weight
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask

    def forward(self, cancer_logits, prostate_mask, needle_mask, ood_mask, label, involvement):
        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            if self.prostate_mask:
                mask &= prostate_mask[i] > 0.5
            if self.needle_mask:
                mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        preds_ood, _ = MaskedPredictionModule()(cancer_logits, ood_mask > 0.5)
        labels = torch.zeros(len(predictions), device=predictions.device)
        for i in range(len(predictions)):
            labels[i] = label[batch_idx[i]]
        labels = labels[..., None]  # needs to match N, C shape of preds
        labels_ood = torch.zeros(len(preds_ood), device=preds_ood.device)
        preds_ood = preds_ood.squeeze()

        loss = torch.tensor(0, dtype=torch.float32, device=predictions.device)
        
        if self.base_loss == "ce":
            loss += nn.functional.binary_cross_entropy_with_logits(
                predictions,
                labels,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=predictions.device
                ),
            )

        elif self.base_loss == "consistency":
            ce = nn.functional.binary_cross_entropy_with_logits(
                predictions,
                label,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=predictions.device
                ),
            )
        
        elif self.base_loss == "obp": # CE + out-of-bounds penalty
            ce = nn.functional.binary_cross_entropy_with_logits(
                predictions,
                labels,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=predictions.device
                ),
            )
            obp = nn.functional.binary_cross_entropy_with_logits(
                preds_ood, 
                labels_ood, 
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=predictions.device
                ),
            )
            loss += ce + (labels_ood.shape[0] / labels.shape[0]) * obp
        
        elif self.base_loss == "inv_mae": # involvement-aware MAE
            if str(involvement.item()).lower() == "nan":
                involvement = 0.0
            
            loss_unreduced = nn.functional.l1_loss(
                predictions, labels * involvement, reduction="none"
            )
           
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()
            loss += nn.functional.l1_loss(predictions.mean(), involvement)

        elif self.base_loss == "inv_mse": # involvement-aware MSE
            if str(involvement.item()).lower() == "nan":
                involvement = 0.0
            
            loss_unreduced = nn.functional.mse_loss(
                predictions, (labels * involvement).float(), reduction="none"
            ).float()
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()
            loss = loss.float()

        elif self.base_loss == "inv_mse_obp": # involvement-aware MSE + out-of-bounds penalty
            if str(involvement.item()).lower() == "nan":
                involvement = 0.0

            loss_unreduced = nn.functional.mse_loss(
                predictions, (labels * involvement).float(), reduction="none"
            ).float()

            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()

            obp = nn.functional.binary_cross_entropy_with_logits(
                preds_ood, 
                labels_ood, 
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=predictions.device
                ),
            )

            loss += (labels_ood.shape[0] / labels.shape[0]) * obp
            loss = loss.float()

        elif self.base_loss == "inv_ce_obp": # involvement-aware CE + out-of-bounds penalty
            if str(involvement.item()).lower() == "nan":
                involvement = 0.0
            involvement = involvement_label_smoothing(label, involvement)

            ce = nn.functional.binary_cross_entropy_with_logits(
                predictions,
                labels * involvement,
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=predictions.device
                ),
            )

            obp = nn.functional.binary_cross_entropy_with_logits(
                preds_ood, 
                labels_ood, 
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=predictions.device
                ),
            )

            loss += ce + (labels_ood.shape[0] / labels.shape[0]) * obp

        elif self.base_loss == "inv_mae_obp": # involvement-aware MAE + out-of-bounds penalty
            if str(involvement.item()).lower() == "nan":
                involvement[0] = 0.0
            
            loss_unreduced = nn.functional.l1_loss(
                predictions, labels * involvement, reduction="none"
            )
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()
            loss += nn.functional.l1_loss(predictions.mean(), involvement)

            obp = nn.functional.binary_cross_entropy_with_logits(
                preds_ood, 
                labels_ood, 
                pos_weight=torch.tensor(
                    self.loss_pos_weight, device=predictions.device
                ),
            )

            loss += (labels_ood.shape[0] / labels.shape[0]) * obp
        
        elif self.base_loss == "gce":
            # we should convert to "two class" classification problem
            loss_fn = BinaryGeneralizedCrossEntropy()
            loss += loss_fn(predictions, labels)
        
        elif self.base_loss == "mae":
            loss_unreduced = nn.functional.l1_loss(
                predictions, labels, reduction="none"
            )
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()
        
        else:
            raise ValueError(f"Unknown base loss: {self.base_loss}")

        return loss


class CancerDetectionSoftValidRegionLoss(CancerDetectionLossBase):
    def __init__(
        self,
        loss_pos_weight: float = 1,
        prostate_mask: bool = True,
        needle_mask: bool = True,
        sigma: float = 15,
    ):
        super().__init__()
        self.loss_pos_weight = loss_pos_weight
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask
        self.sigma = sigma

    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement):
        masks = []
        for i in range(len(cancer_logits)):
            mask = prostate_mask[i] > 0.5
            mask = mask & (needle_mask[i] > 0.5)
            mask = mask.float().cpu().numpy()[0]

            # resize and blur mask
            from skimage.transform import resize

            mask = resize(mask, (256, 256), order=0)
            from skimage.filters import gaussian

            mask = gaussian(mask, self.sigma, mode="constant", cval=0)
            mask = mask - mask.min()
            mask = mask / mask.max()
            mask = torch.tensor(mask, device=cancer_logits.device)[None, ...]

            masks.append(mask)
        masks = torch.stack(masks)

        B = label.shape[0]
        label = label.repeat(B, 1, 256, 256).float()
        loss_by_pixel = nn.functional.binary_cross_entropy_with_logits(
            cancer_logits,
            label,
            pos_weight=torch.tensor(self.loss_pos_weight, device=cancer_logits.device),
            reduction="none",
        )
        loss = (loss_by_pixel * masks).mean()
        return loss


class MultiTermCanDetLoss(CancerDetectionLossBase):
    def __init__(self, loss_terms, weights):
        super().__init__()
        self.loss_terms = loss_terms
        self.weights = weights

    def forward(self, cancer_logits, prostate_mask, needle_mask, ood_mask, label, involvement):
        loss = torch.tensor(0, dtype=torch.float32, device=cancer_logits.device)
        for term, weight in zip(self.loss_terms, self.weights):
            loss += weight * term(
                cancer_logits, prostate_mask, needle_mask, ood_mask, label, involvement
            )
        return loss
    

class MultiHmapMultiTermCanDetLoss(CancerDetectionLossBase):
    def __init__(self, loss_terms, weights):
        super().__init__()
        self.loss_terms = loss_terms
        self.weights = weights

    def forward(self, logits_w, logits_s, prostate_mask, needle_mask, ood_mask, label, involvement):
        loss = torch.tensor(0, dtype=torch.float32, device=logits_w[0].device)
        for term, weight in zip(self.loss_terms, self.weights):
            for i in range(len(logits_w)):
                loss += weight * term(
                    logits_w, logits_s, prostate_mask, needle_mask, ood_mask, label, involvement
                )
        return loss


class BinaryGeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, pred, labels):
        pred = pred.sigmoid()[..., 0]
        labels = labels[..., 0].long()
        pred = torch.stack([1 - pred, pred], dim=-1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, 2).float().to(pred.device)
        gce = (1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()

