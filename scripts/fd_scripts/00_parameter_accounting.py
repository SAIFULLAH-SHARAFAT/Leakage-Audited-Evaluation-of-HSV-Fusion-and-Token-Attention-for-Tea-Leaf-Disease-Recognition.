#!/usr/bin/env python3
"""Parameter accounting for the proposed shared-weight color branch."""
DIMS=[96,192,384,768]; DEPTHS=[2,2,18,2]; MLP_RATIO=4
s=sum(d*n for d,n in zip(DIMS,DEPTHS))
faithful=(1+MLP_RATIO+1)*s  # a:D, b:4D, c:D
minimal=2*s                # a:D, c:D
stem_raw=3*96*4*4+96
stem_sc=4*96*4*4+96
print("sum(depth*dim)             =",s)
print("faithful a/b/c couplings   =",faithful, f"({faithful/1e6:.6f}M)")
print("minimal a/c couplings      =",minimal, f"({minimal/1e6:.6f}M)")
print("raw-HSV patch stem         =",stem_raw, f"({stem_raw/1e6:.6f}M)")
print("sin-cos HSV patch stem     =",stem_sc, f"({stem_sc/1e6:.6f}M)")
print("faithful + sin-cos stem    =",faithful+stem_sc, f"({(faithful+stem_sc)/1e6:.6f}M)")
print("minimal + sin-cos stem     =",minimal+stem_sc, f"({(minimal+stem_sc)/1e6:.6f}M)")
print("vs fresh Stage-4 TLA 3,545,089 params:")
print("  reduction faithful ~=",3545089/(faithful+stem_sc),"x")
print("  reduction minimal  ~=",3545089/(minimal+stem_sc),"x")
