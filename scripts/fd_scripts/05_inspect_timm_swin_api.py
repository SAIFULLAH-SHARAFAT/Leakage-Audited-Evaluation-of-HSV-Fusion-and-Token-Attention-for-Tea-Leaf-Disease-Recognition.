#!/usr/bin/env python3
"""Inspect the installed timm Swin internals before implementing shared-KV hooks."""
import inspect, json
import timm
m=timm.create_model("swin_small_patch4_window7_224.ms_in1k",pretrained=False,num_classes=0)
print("timm",timm.__version__)
print("model",type(m).__module__,type(m).__name__)
print("layers",len(m.layers))
for si,stage in enumerate(m.layers):
    print(f"\nstage {si}: {type(stage).__module__}.{type(stage).__name__}",inspect.signature(stage.forward))
    blocks=getattr(stage,"blocks",None)
    if blocks is None: continue
    b=blocks[0]
    print(" block",type(b).__module__,type(b).__name__,inspect.signature(b.forward))
    print(" block attrs",[a for a in ["norm1","attn","drop_path1","norm2","mlp","drop_path2","ls1","ls2"] if hasattr(b,a)])
    a=b.attn
    print(" attn",type(a).__module__,type(a).__name__,inspect.signature(a.forward))
    print(" attn attrs",[x for x in ["qkv","proj","attn_drop","proj_drop","relative_position_bias_table","relative_position_index","window_size","num_heads","scale"] if hasattr(a,x)])
print("\nFreeze this timm version before writing/running the block adapter.")
