# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .deformable_detr import build as build_deformable

def build_model(args):
    if args.deformable_detr:
        return build_deformable(args)
    else:
        return build(args)