---
annotations_creators:
- expert-generated
language:
- en
license: mit
task_categories:
- image-classification
task_ids:
- multi-class-image-classification
tags:
- agriculture
- plant-pathology
- computer-vision
- tea-leaf
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/Tea_leaf_dataset/train/*/*
  - split: validation
    path: data/Tea_leaf_dataset/val/*/*
  - split: test
    path: data/Tea_leaf_dataset/test/*/*
---

# Data

We do not version image data in this repository. We track only manifests, audit
outputs, and fingerprints, which is enough to verify and rebuild the exact partition
we used.

## Layout

```text
data/
  raw/                   TeaLeafBD release as downloaded      (gitignored)
  tea_leaf_clean/        intermediate build                   (gitignored)
  Tea_leaf_dataset/      FINAL audited dataset                (gitignored)
  manifests/             tracked audit artifacts

data/manifests/
  final_split_manifest.csv   one row per instance: split, class, MD5, canonical id, derived flag
  final_fingerprint.json     counts, thresholds, and the manifest SHA-256
  final_linked_pairs.csv     perceptual near-duplicate pairs found before pruning
  pruned_images.csv          the 79 evaluation images we removed, with measured distances
  aug_params.csv             augmentation parameters recovered from each derivative
  montages/                  the visual evidence we used to choose the thresholds
```

We obtain the images with `scripts/00_fetch_frozen_dataset.py` and verify them with
`scripts/01_verify_frozen_dataset.py`; see the repository README.
