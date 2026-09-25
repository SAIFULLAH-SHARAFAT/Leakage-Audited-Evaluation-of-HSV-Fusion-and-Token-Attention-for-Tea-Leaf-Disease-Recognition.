# Dataset provenance

## Frozen experimental source

Our **experimental source of truth is the public Hugging Face dataset**
`saifullah03/tea-leaf-disease-dataset`, together with the tracked final manifest and
fingerprint. The frozen working set has **7,714 effective instances**:

- train: 6,090
- validation: 816
- test: 808

The manifest lock is:

`d407fdb133ebc95c3b0bc6f815280786bcdef37cd6c3c009192198fd21943d3d`

Every v2 run verifies that lock before training.

## Historical construction record

The TeaLeafBD snapshot we downloaded from Mendeley Data at the start of the project
contained **5,674 images**. We split that snapshot and used training-only augmentation to
construct the processed `tea-leaf701515` working corpus. Leakage and near-duplicate cleaning
then removed 79 evaluation images, leaving 5,595 originals and the frozen 7,714-instance
release above (5,595 + 79 = 5,674).

This is a record of the snapshot we actually used, not a claim that the current upstream
Mendeley version still contains 5,674 images. To reproduce our experiments, use the frozen
Hugging Face release and its commit SHA rather than reconstructing from a mutable upstream
version.

## Leakage scope

The released manifest and verification scripts check file-set identity, exact MD5 overlap,
canonical source-family overlap, confinement of declared derivatives to training, and
cross-partition perceptual near-duplicates at the stated operating point, for originals
and for training derivatives against evaluation images. We describe exactly these checks in
the paper, and we do not broaden them into a claim that every possible semantic
near-duplicate has been detected.
