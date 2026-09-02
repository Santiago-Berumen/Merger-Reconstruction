# Reconstructing Black Hole Merger History from Simulated Gravitational Waves

This pipeline simulates populations of merging black holes, generates the gravitational wave signals their mergers would produce, and then uses Bayesian statistical inference to reconstruct their formation history, testing how well we can recover the truth about an event we can never directly observe, only measure indirectly through noisy data.

This project is a from-scratch, independent reproduction of research I've conducted as an undergraduate researcher with the LIGO Collaboration while at Penn State, rebuilt here with public tools and synthetic data so I could share the methodology openly.

## The problem

When two black holes merge, they form a new, more massive black hole, which can then go on to merge again and again. Over multiple generations, this builds what we call a "family tree" of mergers. These events often are unobservable, and the only evidence we have of any of this is the gravitational wave signal reaching our detectors, which is noisy, indirect, and doesn't come with a label telling you which generation of merger you're looking at.

The core question this pipeline explores: **Given only the noisy signal, how accurately can we reconstruct the true history of the black holes that produced it?**

## Why I built it this way

Every stage of this pipeline involved a real scientific reasoning:

- **Population generation** uses a custom mass-ratio-dependent pairing function and a tapered power-law mass distribution, rather than uniform or naive sampling, because real astrophysical populations aren't uniform; using unrealistic input distributions would make every downstream result meaningless before the inference step even begins.
- **Bayesian nested sampling** is used for reconstruction specifically because the goal is a full posterior distribution with proper uncertainty quantification, so I can report *how confident* the recovery is, not just what it recovered.
- **Validation is a first-class pipeline stage** — I built a systematic accuracy-tracking layer (calibration/PP plots, true-vs-recovered comparisons).

## What I found

Running this pipeline at scale surfaced a genuine, non-obvious result: there's a **tension between mass recovery and spin recovery**, driven by a parameter called `chi_eff` (effective spin). Cases with near-zero effective spin recover mass very accurately but spin poorly; cases with larger spin recover spin better but mass worse. This reflects a real, known degeneracy in how these two parameters imprint on the gravitational wave signal. Catching and correctly interpreting this (rather than assuming something was broken) was one of the more useful outcomes of building out the validation layer.

## Pipeline stages

1. **Merger tree generation** — builds synthetic, multi-generation black hole merger populations using physically-motivated mass and pairing distributions
2. **Signal simulation** — simulates the gravitational wave detection process for each merger event
3. **Bayesian reconstruction** — uses nested sampling to infer the parameters of the merger from the simulated signal, with full posterior uncertainty
4. **Validation** — compares recovered parameters against known ground truth across many runs, using calibration diagnostics to quantify reliability

## Tech stack

- **Python** (numpy, pandas)
- **Bayesian inference / nested sampling** (`dynesty`, `bilby`)
- **Statistical validation** — calibration (PP) plots, credible-interval coverage tracking

## Project structure

```
├── generate.py           # Merger population / synthetic data generation
├── simulate.py           # Gravitational wave signal simulation
├── infer.py              # Bayesian nested-sampling reconstruction
├── validate.py           # Accuracy analysis & calibration plots
├── notebooks/
│   └── end_to_end_demo.ipynb   # Full pipeline walkthrough on a small example
├── results/
│   └── sample plots (PP plot, true-vs-recovered comparison)
└── requirements.txt
```

## Results

*(Sample output plots go here — a true-vs-recovered comparison plot and a PP plot demonstrating calibration accuracy)*

## Contact

Santiago Berumen — [santiagoberumen65@gmail.com]
