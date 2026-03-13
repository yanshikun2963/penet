# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark import _C

# nms directly from C extension (no apex dependency)
nms = _C.nms
