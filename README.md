# Power Surge — Reproducibility Artifact

Companion code for the survey **"Power Surge: A Comprehensive Survey of the Electrical
Impact of Artificial Intelligence Inference in Modern Data Centers,"** by Nishant Mehta
(submitted to *IEEE Communications Surveys & Tutorials*).

`reproduce_dc_inference_power.py` regenerates all figures (Fig. 1–10) and the summary
tables in the paper from the primary data compiled in the script.

## Requirements
- Python 3.9+
- `numpy`, `pandas`, `matplotlib`, `scipy`

```bash
pip install numpy pandas matplotlib scipy
```

## Usage
```bash
python reproduce_dc_inference_power.py
```
This writes `fig1_*.png` … `fig10_*.png` to the current directory and prints the
summary tables to stdout.

## Contents
- `reproduce_dc_inference_power.py` — figure/table generator (single, self-contained script)
- `references.bib` — bibliography used in the manuscript
- `fig1_*.png` … `fig10_*.png` — figures as they appear in the paper (also produced by running the script)

## Notes
- All quantitative inputs (accelerator specs, PUE trends, per-query energy, carbon
  intensities, etc.) are defined inline in the script with their sources noted in comments.
- Figures are produced at the sizes/labels used in the manuscript.

## Citation
Please cite the paper once published; bibliographic details will be added here upon acceptance.

## License
Released under the MIT License (add a `LICENSE` file, or adjust as preferred).
