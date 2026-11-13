# Sample brain MRI slices (demo inputs)
These 7 images are **real structural brain MRI slices** (T1-weighted, skull-stripped) cut from the MNI152 average brain template (average of 152 healthy adults) that ships with the open-source `nilearn` package. They are resized to 224x224 and saved as PNG.

* They are **not Alzheimer's-labelled** and not from a patient. Use them only to test/demonstrate the upload-and-predict app (`python src/demo_app.py`).
* The checkpoints in `models/` were trained on the small 240-image subset in `dataset/processed/`, so predictions on these real slices are NOT meaningful until the models are retrained on a real labelled dataset (see `src/download_real_dataset.py`).
* Axial slices at z = 70, 80, 90, 100, 110, 125 and one coronal slice.
