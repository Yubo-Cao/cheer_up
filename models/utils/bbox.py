import math
from typing import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch
from torch import Tensor, tensor
from torchvision import transforms

cmap = plt.get_cmap("tab20b")


def ellipse_bbox(ellipse: Tensor) -> Tensor:
    """Convert an ellipse to a corner to corner rectangle."""

    if ellipse.ndim == 1:
        cos, sin = torch.cos, torch.sin
        a, b, theta, x, y = ellipse
        ta = a * cos(theta)
        tb = b * sin(theta)
        tc = a * sin(theta)
        td = b * cos(theta)
        dx = (ta**2 + tb**2) ** 0.5
        dy = (tc**2 + td**2) ** 0.5
        return tensor([x - dx, y - dy, x + dx, y + dy])
    else:
        # last dimension is eclipse
        return torch.stack([ellipse_bbox(e) for e in ellipse[..., :]]).reshape(
            ellipse.shape[:-1] + (4,)
        )


def tensor_ellipse(ellipse: Tensor, color: str = "r") -> mpl.patches.Ellipse:
    """Convert an ellipse to a matplotlib ellipse."""

    if ellipse.ndim == 1:
        a, b, theta, x, y = ellipse
        return mpl.patches.Ellipse(
            (x, y),
            a * 2,
            b * 2,
            angle=theta / math.pi * 180,
            fill=False,
            linewidth=1,
            edgecolor=color,
        )
    else:
        # last dimension is eclipse
        return mpl.collections.PatchCollection(
            [tensor_ellipse(e, color) for e in ellipse[..., :]],
            match_original=True,
        )


def tensor_rect(bbox: Tensor, color: str = "r") -> mpl.patches.Rectangle:
    """Convert a bounding box to a matplotlib rectangle."""

    return mpl.patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=1,
        fill=False,
        edgecolor=color,
    )


def _ensure_ax(ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    if ax is None:
        _, ax = plt.subplots()
    return ax


def show_bbox(
    bbox: Tensor,
    ax: Optional[plt.Axes] = None,
    color: Union[str, Iterable[str]] = "r",
    label: Optional[Iterable[str]] = None,
) -> None:
    """Plot bounding box(s) on the image."""

    if bbox.ndim == 1:
        bbox = bbox[None, ...]
    if not isinstance(color, list | Tensor | np.ndarray):
        color = [color] * len(bbox)

    if label is None:
        label = [""] * len(bbox)
    elif not isinstance(label, list):
        label = list(label)

    ax = _ensure_ax(ax)
    try:
        for b, c, l in zip(bbox, color, label, strict=True):
            ax.add_patch(tensor_rect(b, c))
            ax.text(b[0], b[1], l, color=c)
    except ValueError as e:
        raise ValueError(
            "The number of labels (colors) must be the same as the number of bounding boxes."
        )


def show_ellipse(
    ellipse: Tensor, ax: Optional[plt.Axes] = None, color: str = "r"
) -> None:
    """Plot an ellipse on the image."""

    if ellipse.ndim == 1:
        ellipse = ellipse[None, ...]
    ax = _ensure_ax(ax)
    for e in ellipse:
        ax.add_patch(tensor_ellipse(e, color=color))


def show_image(image: Tensor | Image.Image, ax: plt.Axes = None):
    """Show an image."""

    ax = _ensure_ax(ax)
    if isinstance(image, Tensor):
        image = transforms.ToPILImage()(image)
    ax.imshow(image)
    ax.axis("off")


def bbox_corner2center(bbox: Tensor) -> Tensor:
    """Convert a corner to corner bounding box to a center to corner bounding box."""

    return torch.stack(
        [
            (bbox[..., 0] + bbox[..., 2]) / 2,
            (bbox[..., 1] + bbox[..., 3]) / 2,
            bbox[..., 2] - bbox[..., 0],
            bbox[..., 3] - bbox[..., 1],
        ],
        dim=-1,
    )


def bbox_center2corner(bbox: Tensor) -> Tensor:
    """Convert a center to corner bounding box to a corner to corner bounding box."""

    return torch.stack(
        [
            bbox[..., 0] - bbox[..., 2] / 2,
            bbox[..., 1] - bbox[..., 3] / 2,
            bbox[..., 0] + bbox[..., 2] / 2,
            bbox[..., 1] + bbox[..., 3] / 2,
        ],
        dim=-1,
    )


def bbox_corner_width2corner(bbox: Tensor) -> Tensor:
    """Convert a corner to width bounding box to a corner to corner bounding box."""

    return torch.stack(
        [
            bbox[..., 0],
            bbox[..., 1],
            bbox[..., 0] + bbox[..., 2],
            bbox[..., 1] + bbox[..., 3],
        ],
        dim=-1,
    )


def test_bbox_conversion():
    bbox = torch.rand(10, 4)
    assert torch.allclose(bbox, bbox_center2corner(bbox_corner2center(bbox)))


def anchors(data, sizes, ratios) -> Tensor:
    """Generate anchors for each pixel of the image in the data. For each
    pixel, (sizes[0], ratios) + (ratios[0], sizes) are generated. Notice the
    duplicated (sizes[0], ratios[0]) will be removed.

    The result will be a tensor of shape (1, (len(sizes) + len(ratios) - 1) * width * height, 4)"""

    width, height = data.shape[-2:]
    n_sizes, n_ratios = len(sizes), len(ratios)
    boxes_per_pixel = n_sizes + n_ratios - 1

    device = data.device
    sizes = tensor(sizes, device=device)
    ratios = tensor(ratios, device=device)

    center_h, center_w = [
        s * (torch.arange(0, n, device=device) + 0.5)
        for n, s in zip((height, width), (1 / height, 1 / width))
    ]
    center_w, center_h = [
        c.flatten() for c in torch.meshgrid(center_w, center_h, indexing="ij")
    ]
    w = (
        torch.cat(
            (
                sizes * torch.sqrt(ratios[0]),
                sizes[0] * torch.sqrt(ratios[1:]),
            )
        )
        * height
        / width
    )
    h = torch.cat(
        (
            sizes / torch.sqrt(ratios[0]),
            sizes[0] / torch.sqrt(ratios[1:]),
        )
    )

    anchors = torch.stack((-w, -h, w, h)).T.repeat(height * width, 1) / 2
    return (
        torch.stack((center_w, center_h, center_w, center_h), dim=1).repeat_interleave(
            boxes_per_pixel, dim=0
        )
        + anchors
    ).unsqueeze(0)


def bbox_scale(bbox: Tensor, image: Tensor | Image.Image):
    """Scale a bounding box to the size of the image."""

    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)
    return bbox * tensor(image.shape[-2:]).flip(0).repeat(2)


def box_area(box: Tensor) -> Tensor:
    """Compute the area of a bounding box."""

    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])


def iou(a: Tensor, b: Tensor) -> Tensor:
    """Calculate the intersection over union (Jaccard Coefficient) of 2 bboxes"""

    aa, ab = box_area(a), box_area(b)
    # upper left corner (in cartesian plane, where y increases downward (as in image))
    inter_ul = torch.max(a[..., :2], b[..., :2])
    # lower right corner
    inter_lr = torch.min(a[..., 2:], b[..., 2:])
    inter = (inter_lr - inter_ul).clamp(min=0)
    inter_area = inter[..., 0] * inter[..., 1]
    # 𝐉(a, b) = a ∩ b / a ∪ b
    union_area = aa + ab - inter_area
    return inter_area / union_area


def assign_gt(anchors: Tensor, gts: Tensor, iou_threshold: float = 0.5) -> Tensor:
    """Assign ground truth to each bounding box. If a bounding box has an
    intersection over union with a ground truth greater than iou_threshold,
    then the ground truth is assigned to the bounding box. Otherwise, the
    bounding box is assigned to the background class."""

    ious = iou(anchors.unsqueeze(1), gts.unsqueeze(0))
    anchor2gt = torch.full((len(anchors),), -1)

    max_ious, max_ious_idxs = ious.max(dim=1)
    anchor_idxs = torch.nonzero(max_ious >= iou_threshold).flatten()
    gt_idxs = max_ious_idxs[max_ious >= iou_threshold]
    anchor2gt[anchor_idxs] = gt_idxs

    for _ in range(len(gts)):
        max_idx = ious.argmax()
        anchor_idx, gt_idx = torch.div(
            max_idx, len(gts), rounding_mode="trunc"
        ), max_idx % len(gts)
        anchor2gt[anchor_idx] = gt_idx
        ious[anchor_idx, :] = ious[:, gt_idx] = -1
    return anchor2gt


def bbox_offset(anchors, gts, eps=1.0e-6):
    """Calculate the offset between the anchors and the ground truth bounding
    boxes."""

    anchors = bbox_corner2center(anchors)
    gts = bbox_corner2center(gts)
    wh = anchors[..., 2:]
    oxy = 10 * (gts[..., :2] - anchors[..., :2]) / wh
    owh = 5 * torch.log(eps + gts[..., 2:] / wh)
    return torch.cat((oxy, owh), axis=-1)


def bbox_recover(anchors, offsets):
    """Recover the bounding box from the offset."""

    anchors = bbox_corner2center(anchors)
    wh = anchors[..., 2:]
    oxy = offsets[..., :2] / 10 * wh + anchors[..., :2]
    owh = torch.exp(offsets[..., 2:] / 5) * wh
    return bbox_center2corner(torch.cat((oxy, owh), axis=-1))


def assign_label(anchors, labels):
    """Assign labels to anchors. For anchors that are not assigned to a
    ground truth, they are assigned to the background class.

    :param anchors: (batch_size)? x num_anchors x 4
    :param labels: (batch_size)? x num_anchors x 5
        where the last dimension is (class, bbox)

    :return: (
        offset: (batch_size)? x num_anchors x 4,
        mask: (batch_size)? x num_anchors x 4, where background class is 0
        classes: (batch_size)? x num_anchors, where 0 is the background class
    )
    """

    if labels.ndim > 2:
        return (
            torch.stack(x)
            for x in zip(*[assign_label(anchors, label) for label in labels])
        )

    anchors2gt = assign_gt(anchors, labels[..., 1:])  # first column is class
    not_bg = anchors2gt >= 0
    masks = not_bg.float().unsqueeze(-1).repeat(1, 4)

    classes = torch.full((len(anchors),), 0)
    classes[not_bg] = labels[anchors2gt[not_bg], 0].long() + 1

    anchors2bbox = torch.zeros((len(anchors), 4))
    anchors2bbox[not_bg] = labels[anchors2gt[not_bg], 1:].float()

    offsets = bbox_offset(anchors, anchors2bbox) * masks
    return (offsets, masks, classes)


def nms(bboxes, scores, iou_threshold=0.5):
    """Non-maximum suppression.

    :param bboxes: (batch_size)? x num_bboxes x 4
    :param scores: (batch_size)? x num_bboxes
    :param iou_threshold: the threshold for the intersection over union

    :return: (batch_size)? x indices of bboxes that are not suppressed
    """

    idxs = scores.argsort(descending=True, dim=-1)
    keep = []
    while len(idxs) > 0:
        keep.append(idxs[0])
        if len(idxs) == 1:
            break
        ious = iou(bboxes[idxs[0]].unsqueeze(0), bboxes[idxs[1:]])
        idxs = idxs[1:][ious <= iou_threshold]
    return tensor(keep)


def detect(anchors, cls_probs, offsets, confidence_threshold=0.01, nms_threshold=0.5):
    """Give anchors, class probabilities, and offsets, return the bounding
    boxes, their class probabilities, and their class labels. Further, apply
    non-maximum suppression to the bounding boxes.

    :param anchors: (batch size)? x (number of anchors) x 4
    :param cls_probs: (batch size)? x (number of classes) x (number of anchors)
    :offsets: (batch size)? x (number of anchors) x 4
    :return: (batch size)? x (number of bounding boxes) x 6, where
        the last dimension is (class label, class probability/confidence, bounding box)
    """

    if cls_probs.ndim > 2 and offsets.ndim > 2:  # support batched inputs
        return (
            torch.stack(x)
            for x in zip(
                *[
                    detect(anchors, cls_prob, offset)
                    for cls_prob, offset in zip(cls_probs, offsets)
                ]
            )
        )

    confidence, anchor_idxs = cls_probs.max(
        dim=-2
    )  # operate on the columns, i.e., anchors
    predict_bboxes = bbox_recover(anchors, offsets)
    keep = nms(predict_bboxes, confidence, nms_threshold)

    # set background class to -1 (not in keep or confidence < confidence_threshold)
    anchor_idxs[
        torch.cat((keep, torch.arange(len(anchors))),).unique(
            return_counts=True
        )[1]
        == 1
    ] = -1
    anchor_idxs[confidence < confidence_threshold] = -1

    return torch.cat(
        (
            anchor_idxs.unsqueeze(-1),
            confidence.unsqueeze(-1),
            predict_bboxes,
        ),
        axis=-1,
    )


def show_result(img, results, class_names: list[str], ax: Optional[plt.Axes] = None):
    """
    Show the detection results.

    :param img: (height, width, 3)
    :param results: (number of bounding boxes) x 6, where the last dimension
        is (class label, confidence, bounding box)
    :param class_names: list of classes
    :param ax: the axes to use to plot
    """

    ax = _ensure_ax(ax)
    show_image(img, ax=ax)
    color_map = dict(
        zip(
            class_names,
            cmap(np.linspace(0, 1, len(class_names) + 1)),
        )
    )
    class_names.insert(0, "background")
    results = results[results[:, 0] > 0]

    labels = [f"{class_names[int(result[0])]}: {result[1]:.2f}" for result in results]
    colors = [color_map[class_names[int(result[0])]] for result in results]
    bboxes = results[:, 2:]
    bboxes = bbox_scale(bboxes, img)

    show_bbox(
        bboxes,
        label=labels,
        color=colors,
        ax=ax,
    )
