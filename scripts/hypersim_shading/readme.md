# Hypersim diffuse shading dataset

Builds `assets/datasets/marigold_train_shading` following the Marigold V1.1
IID-lighting split: diffuse shading is Hypersim's linear-RGB
`diffuse_illumination`, clamped to `>= 0.0` (with optional percentile
normalization to `[0, 1]`) and stored as HWC float32 `.npy`. Only the diffuse
illumination files are downloaded; RGB images are hard-linked from the
preprocessed depth dataset (`marigold_train`), which must exist first. Finished
scenes are skipped on rerun.

```bash
bash scripts/hypersim_shading/download_and_preprocess_hypersim_shading.sh