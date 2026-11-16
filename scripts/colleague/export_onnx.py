"""
Export the colleague's BladderTouchDynamicCrop checkpoint to ONNX for cv2.dnn.

Only the weights are loaded (torch.load(weights_only=True) into a torchvision
EfficientNet-B0 with a 2-class head); none of their code is imported. The input is a
fixed 1x3x224x224 batch, because OpenCV 4.6 cannot load the export with dynamic axes.
The script checks that cv2.dnn and torch agree on random inputs.

Usage (training venv): export_onnx.py <best.pt> <out.onnx>
"""
import argparse

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_b0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('checkpoint', help='runs/<name>/best.pt')
    parser.add_argument('out', help='output .onnx')
    args = parser.parse_args()

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    model.eval()
    torch.onnx.export(model, (torch.zeros(1, 3, 224, 224),), args.out, opset_version=12,
                      input_names=['image'], output_names=['logits'], dynamo=False)

    net = cv2.dnn.readNetFromONNX(args.out)
    worst = 0.0
    for seed in range(3):
        x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(seed))
        with torch.inference_mode():
            ref = model(x).softmax(1)[:, 1].numpy()
        net.setInput(x.numpy())
        z = net.forward()
        got = np.exp(z[:, 1]) / np.exp(z).sum(1)
        worst = max(worst, float(np.abs(ref - got).max()))
    print(f'{args.out}: cv2.dnn vs torch p_touch max diff {worst:.2e}')


if __name__ == '__main__':
    main()
