# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
from .label_classifier import *
from .detr_multi_classifier import *

def build_label_classifier(args):
    if args.type=="multi":
        return DetrClassifier(args)
    elif args.type=="linear":
        return LinearClassifier(args)
    elif args.type=="dict":
        return DictClassifier(args)
    else:
        raise KeyError

