# NEURON Voltage Clamp Lab (HH single-compartment)

This repository hosts an interactive Jupyter/ipywidgets lab that runs a voltage-clamp simulation
using the NEURON simulator on a single-compartment Hodgkin–Huxley model.

## Shareable link (runs in the browser)
GitHub Pages **cannot** execute Python/NEURON. The usual way to share a "website-like" link is **Binder**:

1. Push this repo to GitHub
2. Replace `<USER>` and `<REPO>` in the badge URL below
3. Send your collaborator the Binder link

**Binder launch (edit the URL after you push):**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/<USER>/<REPO>/HEAD?labpath=notebooks%2FVoltageClampLab.ipynb)

> If Binder ever fails to build due to upstream NEURON wheels/OS changes, use the local install instructions below.

## Run locally (recommended for reliability)
### Option A: pip + venv
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
jupyter lab
```
Then open: `notebooks/VoltageClampLab.ipynb` and run the cell.

### Option B: conda
```bash
conda env create -f environment.yml
conda activate neuron-lab
jupyter lab
```

## Project structure
- `src/lab.py` – simulation + widget UI (`build_ui()`)
- `notebooks/VoltageClampLab.ipynb` – student-facing launcher notebook
- `requirements.txt` – dependencies (Binder + local)
- `runtime.txt` – pins Python version for Binder

## Notes / fixes applied vs. the original Colab cells
- Removed Colab-only `!pip install ...` magic (dependencies live in `requirements.txt`)
- Reset button now matches default slider values
- Uses fresh NEURON `Vector()` objects each run to avoid recording/link glitches in repeated runs
- Current plot y-limits auto-scale for readability (instead of fixed ±6 nA)
