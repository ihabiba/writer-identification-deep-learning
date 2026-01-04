# Dataset Structure

The dataset is not included in this repository.

Expected structure:

```
dataset/
├── train/
│   ├── 01_*.png
│   ├── 02_*.png
│   └── ...
└── test/
    ├── 01_*.png
    ├── 02_*.png
    └── ...
```

- Filenames must start with a two-digit writer ID.
- Images must be grayscale handwritten pages.
- The preprocessing and inference pipeline assumes this structure.
