# Dataset

`processed/` holds a 240-image subset (160 train / 40 val / 40 test, 40/10/10 per class).
**Provenance warning:** these images look synthetic (uniform ellipse "skull", two dark blobs for ventricles, tick-mark ring), not real MRI slices, and 35/40 test images are near-copies of train images (see `results/near_duplicate_audit.json`). Do NOT describe them as a real public MRI dataset unless you can name the source.

For the final submission use a real 4-class Alzheimer MRI dataset (e.g. Kaggle "Alzheimer's Dataset (4 class of Images)", or OASIS/ADNI) and build the split with:

    python src/prepare_real_dataset.py --src /path/to/dataset --max_per_class 1500

Expected source layout: `<src>/MildDemented/ ModerateDemented/ NonDemented/ VeryMildDemented/`.
Cite the dataset source, licence, and note that public Kaggle versions may contain augmented/oversampled images from the same subjects.

## Real MRI images
* `sample_mri/` now contains real brain MRI slices (MNI152 template, unlabelled) for demo uploads - see `sample_mri/README.md`.
* For real labelled Alzheimer data run `python src/download_real_dataset.py` (Hugging Face `Falah/Alzheimer_MRI`, Apache-2.0) and then `src/prepare_real_dataset.py`.
