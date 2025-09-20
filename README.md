## Carbon-Efficient Internet Video Streaming (CEVS)

This repository contains the codebase for the implementation of **CEVS**.

The strategies described in the paper are implemented in [strategy.py](strategy.py).

Since the coordination server and evaluation components include dataset-specific information, we are withholding them for now to avoid redistributing the dataset. A sanitized version, with data masking applied, will be released shortly.

### Prepare data

1. Follow the schemas provided in [historical_schema.json](historical_schema.json) and [forecast_schema.json](forecast_schema.json).
2. Place the raw dataset files in the `dataset` folder and rename them using the following formats:
    - `hist_co2_moer_<region>_<start_timestamp>>_<end_timestamp>>`
    - `fore_co2_moer_<region>_<start_timestamp>_<end_timestamp>`
3. Run the script [raw_dataset_to_npy.py](raw_dataset_to_npy.py)
 to generate the `.npy` files required by [strategy.py](strategy.py).
