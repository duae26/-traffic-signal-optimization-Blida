# Hybrid Predictive Meta-Heuristic Framework for Traffic Signal Optimization

**M2 Thesis — University of Yahia Fares Medea — 2025-2026**

A hybrid framework combining LSTM-based traffic prediction with
meta-heuristic optimization (WOA, HHO, GA) for dynamic traffic
signal control applied to a real 9-intersection network in
Blida, Algeria.

---

## Key Results

| Method        | Low    | Medium | Peak   |
| ------------- | ------ | ------ | ------ |
| GA (reactive) | -11.1% | -7.7%  | -19.1% |
| WOA + LSTM    | +18.8% | +27.0% | +18.8% |
| HHO + LSTM    | +27.9% | +38.2% | +34.6% |

Improvement relative to fixed-time baseline (mean over 5 runs).

---

## Requirements

- Python 3.10+
- SUMO 1.25.0
- PyTorch 2.0
- See requirements.txt for full list

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

**1. Run all experiments (baseline + WOA + HHO + GA):**

```bash
python run_experiment.py
```

**2. Run statistical validation (5 independent runs):**

```bash
python run_5x.py
```

**3. Generate all figures and Excel table:**

```bash
python generate_results.py
```

---

## Project Structure
