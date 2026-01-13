# NACRA — Longitudinal Breast MRI Response Assessment

⚠️ **Status:** This repository is under active development. Interfaces, prompts, and folder conventions may change.

NACRA is a lightweight longitudinal reporting pipeline that converts sequential breast MRI report descriptions into a final **structured report JSON** aligned to a predefined template

For all specific prompts, please refer to the article's **Supplementary Materials**.

LightRAG is used as an **external retrieval service**. This repository only sends HTTP requests to the LightRAG `/query` endpoint.


## Software Requirements

- Python **3.11+** recommended 
- No special hardware required


## Configuration

### LightRAG URL

This repo sends queries to a LightRAG endpoint.

- Default in code (current): `http://localhost:9621/query`
- Recommended: configure via environment variable:

```bash
export LIGHTRAG_URL="http://localhost:9621/query"
```

> Note: If your current `main.py` / `tools/lightrag.py` still hard-codes the URL, update them to read `LIGHTRAG_URL` from the environment.



## Repository Structure

```
NACRA/
├── main.py                      # Entry point: loads per-patient JSON and runs the pipeline
├── report_processor.py           # Multi-step processing (lesion list / diagnosis / response) + validations
├── recist_eval.py                # RECIST-like response evaluation helper
├── tools/
│   └── lightrag.py               # Thin HTTP client to call LightRAG /query
├── utils/
│   ├── parser.py                 # Baseline detection helper(s)
│   └── tokentracker.py           # Token estimation using tiktoken
├── shots/                        # Prompt configurations (zero/one)
│   ├── birads.json
│   ├── diag.json
│   ├── lesion_list.json
│   ├── recist.json
│   └── structured_report.json
├── template/                     # JSON templates used for structure/validation
│   ├── lesion_list_template.json
│   └── structured_report_template.json
├── records_updated/              # Input folder (recommended): merged/cleaned patient JSONs
├── raw_data/                     # Optional input folder (if you keep original raw files here)
├── lesion_lists/                 # Optional cache/history (per patient)
├── structured_reports/           # Output folder: final structured report JSONs
├── requirements.txt
└── LICENSE.txt
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```
