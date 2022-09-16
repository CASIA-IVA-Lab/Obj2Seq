import torch
from collections import OrderedDict
import json

src_ckpt = torch.load("r50_deformable_detr-checkpoint.pth")["model"]
json_file = json.load(open("/home/coco/annotations/instances_val2017.json"))
cls_idx = torch.as_tensor([a["id"] for a in json_file["categories"]])

new_ckpt = OrderedDict()
for key, value in src_ckpt.items():
    new_value = value
    new_key = key.replace("backbone.0.", "backbone.")
    new_key = new_key.replace("reference_points", "object_decoder.reference_points")
    if "query_embed" in new_key:
        new_ckpt["transformer.object_decoder.position_patterns.weight"] = value[:, 256:]
        new_ckpt["transformer.object_decoder.position.weight"] = value[:, :256]
        continue
    elif "transformer.encoder" in new_key:
        new_key = new_key.replace("transformer.encoder.layers", "transformer.encoder.encoder_layers")
        new_key = new_key.replace(".linear", ".ffn.linear")
        new_key = new_key.replace(".norm2", ".ffn.norm2")
    elif "transformer.decoder" in new_key:
        new_key = new_key.replace("transformer.decoder.layers", "transformer.object_decoder.object_decoder_layers")
        new_key = new_key.replace(".linear", ".ffn.linear")
        new_key = new_key.replace(".norm3", ".ffn.norm2")
    elif "class_embed" in new_key:
        new_key = new_key.replace("class_embed.", "transformer.object_decoder.detect_head.").replace(".weight", ".class_embed.classifier.weight").replace(".bias", ".class_embed.classifier.bias")
        new_value = new_value[cls_idx]
    elif "bbox_embed" in new_key:
        new_key = new_key.replace("bbox_embed.", "transformer.object_decoder.detect_head.").replace(".layers.", ".bbox_embed.layers.")
    new_ckpt[new_key] = new_value

torch.save({"model": new_ckpt}, "new_ckpt.pth")
import pdb;pdb.set_trace()
