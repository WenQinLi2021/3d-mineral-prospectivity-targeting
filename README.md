# 3D Mineral Prospectivity Targeting

This repository contains the analysis code and configuration used in the accompanying study on 3D mineral prospectivity targeting from sparse mineralization labels.

## Data access

The repository does not include the underlying geophysical data, source records, or complete 3D geological-mineralization grid. These materials are subject to data-owner restrictions. Researchers may contact the corresponding author, Yiping Tian (yptian@cug.edu.cn), to request access; any release requires approval from the data owner.

The code expects the authorized input files listed in `config.yaml` under `data/`. Those input files must be obtained separately. Analysis outputs are written under `outputs/` and are not part of this repository.

## Environment

Use Python 3.10 or newer and install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Running the audited pipeline

After placing authorized inputs at the paths configured in `config.yaml`, run:

```bash
python run_pipeline.py --config config.yaml
```

The pipeline performs the audit, modeling, target delineation, and reporting stages. `rerun_exact_pv.py` provides the exact prediction-volume rerun used for the reported evaluation and requires the corresponding audited intermediate files.

## Scope

The code provides the computational workflow. It does not provide permission to access or redistribute restricted input data. Reported results depend on the authorized study data and the input provenance described in the article.
