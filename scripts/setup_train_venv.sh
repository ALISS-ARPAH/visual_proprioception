#!/bin/bash
# Create the PyTorch venv used to train the cut-depth model.
# Usage: scripts/setup_train_venv.sh [venv_dir]   (default ~/aliss_runs/contact_venv)
set -euo pipefail
VENV=${1:-$HOME/aliss_runs/contact_venv}
HERE=$(cd "$(dirname "$0")" && pwd)

# System site packages keep rclpy, rosbag2_py and the apt OpenCV (cv2.dnn runs the ONNX models).
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install -r "$HERE/requirements-train.txt"

"$VENV/bin/python" -I -c "
import cv2, numpy, torch
assert numpy.__version__.startswith('1.'), 'numpy must stay < 2 for the apt OpenCV'
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      '| numpy', numpy.__version__, '| cv2', cv2.__version__)
"
