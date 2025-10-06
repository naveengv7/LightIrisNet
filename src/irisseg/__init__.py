# Re-export key APIs for convenience
from ._train_impl import (
    set_seed, imread_rgb, imread_mask, imread_edge, gaussian_blur_soft_boundary,
    distance_transform_loss_weight, signed_dt_from_mask, infer_dataset_from_id,
    collate_pad, IrisSegDataset, ResNet50Encoder, MNetV3Encoder, ASPP, ASPP_dw,
    DecoderDeepLabV3Plus, PupilRefine, IrisNetDeepLab, DiceLoss,
    TverskyFocalLoss, UncertaintyWeights, dice_score_bin, e1_error,
    pupil_inside_iris_loss, sobel_edges, edge_mask_consistency,
    train_one_epoch, evaluate, save_checkpoint
)

from ._test_impl import (
    largest_cc, refit_ellipse, rasterize_ellipse_from_params, apply_containment, predict
)
