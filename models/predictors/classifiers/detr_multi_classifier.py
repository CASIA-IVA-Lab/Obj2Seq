# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DetrClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_layers = args.num_layers
        if args.num_layers > 0:
            self.feature_linear = nn.ModuleList([nn.Linear(args.hidden_dim, args.hidden_dim) for i in range(args.num_layers)])
        else:
            self.feature_linear = None
        self.classifier = nn.Linear(args.hidden_dim, 80)
        self.reset_parameters(args.init_prob)

    def reset_parameters(self, init_prob):
        prior_prob = init_prob
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.classifier.bias.data, bias_value)

    def forward(self, x, class_vector=None, cls_idx=None):
        # x: bs,cs,(nobj,)d
        # class_vector: bs,cs,d
        if self.feature_linear is not None:
            for i in range(self.num_layers):
                x = F.relu(self.feature_linear[i](x)) if i < self.num_layers - 1 else self.feature_linear[i](x)
        return self.classifier(x)
