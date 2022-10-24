from tkinter.messagebox import askyesno

from utils.bbox import *

def assert_yes(message):
    answer = askyesno("Test", message)
    assert answer, message


def test_draw():
    box = tensor([0.5, 0.5, 1.0, 1.0])
    ellipse = tensor([0.5, 0.5, 0.0, 0.5, 0.5])

    _, ax = plt.subplots()
    show_image(torch.rand(3, 5, 5), ax=ax)
    show_bbox(box, ax=ax)
    show_ellipse(ellipse, ax=ax)
    ax.set_aspect("equal")
    plt.show(block=False)
    assert_yes("See a random image and a box and an ellipse?")



def test_batch_draw():
    boxes = tensor([[0.5, 0.5, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]])
    ellipses = tensor(
        [[2, 1, math.pi / 3, 1, 1], [1.0, 2.0, math.pi / 3, 1.0, 1.0]],
    )  # oblique ecllipse works
    ellipse_bboxes = ellipse2bbox(ellipses)

    _, ax = plt.subplots()
    show_image(torch.rand(3, 5, 5), ax=ax)
    show_bbox(boxes, ax=ax)
    show_bbox(ellipse_bboxes, ax=ax, color="w")
    show_ellipse(ellipses, ax=ax, color="b")
    ax.set_aspect("equal")
    plt.show(block=False)
    assert_yes("See a random image and 4 boxes and two ellipses? Each ellipse is in a box too.")


def test_anchors():
    data = torch.rand(2, 3, 5, 5)
    sizes = [0.1, 0.2]
    ratios = [0.5, 1.0, 2.0]
    assert anchors(data, sizes, ratios).shape == (
        2,  # batch size
        5 * 5 * 4,  # bpp * pixels
        4,  # 4
    )  # channels doesn't matter.


def test_iou():
    a = torch.tensor([[0, 0, 1, 1], [0, 0, 2, 2]])
    b = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    assert torch.allclose(iou(a, b), torch.tensor([1.0, 0.25]))
    a = torch.tensor([[[0, 0, 1, 1], [0, 0, 2, 2]], [[0, 0, 1, 1], [0, 0, 1, 1]]])
    b = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1]]])
    assert torch.allclose(iou(a, b), torch.tensor([[1.0, 0.25], [1.0, 1.0]]))


def test_assign_gt():
    anchors = torch.tensor([[0, 0, 1, 1], [0, 0, 2, 2]])
    gts = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    assert torch.allclose(assign_gt(anchors, gts), torch.tensor([0, 1]))


def test_bbox_recover():
    anchors = torch.tensor([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=torch.float32)
    offsets = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
    assert torch.allclose(bbox_recover(anchors, offsets), anchors)


def test_detect():
    anchors = torch.tensor(
        [[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]], dtype=torch.float32
    )
    cls_probs = torch.tensor([[0, 0, 0], [0.1, 0.9, 0], [0.9, 0.1, 0]])
    offsets = torch.zeros((3, 4))
    assert torch.allclose(
        detect(anchors, cls_probs, offsets),
        torch.tensor(
            [
                [2, 0.9, 0, 0, 1, 1],
                [1, 0.9, 0, 0, 2, 2],
                [-1, 0.0, 0, 0, 3, 3],  # background class
            ]
        ),
    )


def test_scale_bbox():
    image = torch.rand(3, 5, 5)
    bbox = torch.rand(2, 4)
    assert torch.allclose(bbox_scale(image, bbox), bbox * 5)
