# Responsible and Explainable AI Assignment No. 2

**Name:** Zain Ul Wahab  
**Roll No.:** 22I-0491

## Project Overview

This repository contains the notebooks and code for Responsible and Explainable AI Assignment No. 2. The assignment is organized into five notebook parts, each focusing on a different stage of the workflow:

- `part1.ipynb`: Initial setup and baseline exploration.
- `part2.ipynb`: Data processing and early analysis.
- `part3.ipynb`: Model training and evaluation.
- `part4.ipynb`: Mitigation techniques and fairness-focused experiments.
- `part5.ipynb`: Guardrail pipeline with layered moderation, calibration, and threshold analysis.

The supporting Python module `pipeline.py` is used by Part 5 for the moderation pipeline implementation.

## Environment

- Python version: 3.14.3
- GPU used: Nvidia RTX 5070

## Requirements

Install the project dependencies using the requirements file included in the repository:

```bash
pip install -r requirements.txt
```

If you prefer using a virtual environment, create and activate it first, then install the requirements file inside that environment.

## How to Run

1. Create and activate a Python environment.
2. Install dependencies from `requirements.txt`.
3. Open the notebooks in order if you want to reproduce the full assignment flow:
	- `part1.ipynb`
	- `part2.ipynb`
	- `part3.ipynb`
	- `part4.ipynb`
	- `part5.ipynb`
4. Run the notebook cells from top to bottom.

For Part 5, make sure `pipeline.py` is in the same project directory as the notebook so the import works correctly.

## How to Reproduce

To reproduce the results in this repository:

1. Clone or open the project folder.
2. Set up a Python environment with Python 3.14.3.
3. Install dependencies using `requirements.txt`.
4. Ensure the dataset files are present in the `dataset/` folder.
5. Run the notebooks in sequence, especially Part 4 before Part 5 because Part 5 depends on the Part 4 checkpoint.
6. Execute the cells in `part5.ipynb` to generate the moderation pipeline outputs and visualizations.

## Notes

- The dataset used for the assignment is stored in the `dataset/` folder.
- Part 5 generates plots and evaluation summaries based on 1,000 sampled examples.
- If you rerun Part 5, make sure the Part 4 checkpoint directory exists before starting the notebook.