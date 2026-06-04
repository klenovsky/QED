# Quantum Electrodynamics Explorer

A Streamlit app that introduces a minimal quantum-electrodynamics model: one quantized cavity mode interacting with one two-level atom through the Jaynes–Cummings Hamiltonian.

## Included panels

- Quantized light states (Fock, coherent, thermal-like)
- Atom–field exchange dynamics
- Vacuum Rabi oscillations
- Collapse and revival for a coherent field
- Detuning scan heatmap

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

This repository is ready to upload to GitHub and deploy on Streamlit Community Cloud with `app.py` as the main entry point.


This version includes an additional open-system Lindblad panel with cavity loss, atomic relaxation, and pure dephasing.


## GIF export

Animated panels in the app can be exported directly as GIF files. The export is generated from the current numerical data and does not require Chrome or Kaleido.
