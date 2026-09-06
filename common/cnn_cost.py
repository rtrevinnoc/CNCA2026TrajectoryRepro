import os

import cv2
import numpy as np
import torch

from common.cnn_model import WaferCNN

WEIGHTS_PATH = os.environ.get(
    "CNCA_CNN_WEIGHTS",
    os.path.join(os.path.dirname(__file__), "wafer_cnn.pth"))

LABELS = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
          'Loc', 'Near-full', 'Random', 'Scratch']
PLAUSIBLE = {'none', 'Scratch', 'Edge-Loc', 'Random'}

_cnn = None


def cnn():
    global _cnn
    if _cnn is None:
        _cnn = WaferCNN(num_classes=9)
        _cnn.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        _cnn.eval()
    return _cnn


def classify(wafer_map):
    resized = cv2.resize(wafer_map.astype(np.float32), (64, 64),
                         interpolation=cv2.INTER_NEAREST)
    with torch.no_grad():
        logits = cnn()(torch.from_numpy(resized)[None, None])
        probs = torch.softmax(logits, dim=1)[0].numpy()
    k = int(probs.argmax())
    return probs, LABELS[k], float(probs[k])


def map_cost(wafer_map):
    probs, label, conf = classify(wafer_map)
    p_none = float(probs[LABELS.index('none')])
    p_ood = float(sum(probs[LABELS.index(c)] for c in LABELS
                      if c not in PLAUSIBLE))
    return (1.0 - p_none) + p_ood, dict(label=label, conf=conf,
                                        p_none=p_none, p_ood=p_ood)


def dies_to_map(dies, die_l, wafer_r, status):
    n = int(np.ceil(2 * wafer_r / die_l)) + 2
    wm = np.zeros((2 * n, 2 * n), dtype=int)
    index = {d: k for k, d in enumerate(dies)}
    for i in range(-n, n):
        for jj in range(-n, n):
            cx, cy = i * die_l, jj * die_l
            corners = [(cx - die_l / 2, cy - die_l / 2), (cx + die_l / 2, cy - die_l / 2),
                       (cx - die_l / 2, cy + die_l / 2), (cx + die_l / 2, cy + die_l / 2)]
            if all(np.sqrt(x ** 2 + y ** 2) < wafer_r for x, y in corners):
                k = index.get((cx, cy))
                if k is not None and k < len(status):
                    wm[jj + n, i + n] = status[k]
    return wm
