import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import MyD2l as d2l


def show_bboxes(axes, bboxes, labels=None, colors=None, show=True):
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
    if show:
        plt.show()


def test_rect(img):
    fig = plt.imshow(img)
    fig.axes.add_patch(d2l.bbox_to_rect([-60, 45, 378, 516], 'blue'))
    plt.show()


if __name__ == '__main__':
    img = plt.imread('E:/Programming/jupyter_workspace/data/test_pic.jpg')
    # test_rect(img)
    h, w = img.shape[0:2]

    print('height%d, width:%d' % (h, w))
    X = torch.rand(1, 3, h, w)
    Y = d2l.multi_box_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    print('Y.shape: ', Y.shape)

    boxes = Y.view(h, w, 5, 4)
    print('boxes[250, 250, 0, :]: ', boxes[250, 250, 0, :])

    d2l.set_figsize()
    fig = plt.imshow(img)
    print(boxes[1800, 1300, :, :])
    show_bboxes(fig.axes, boxes[1800, 1300, :, :], ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
                                  's=0.75, r=0.5'])

    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                                 [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                            [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                            [0.57, 0.3, 0.92, 0.9]])
    fig = plt.imshow(img)
    bbox_scale = torch.FloatTensor((w, h, w, h))
    show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k', show=False)
    show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
