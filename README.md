# smcpaper
# SMC-Sparsity: Sequential Monte Carlo for High-Sparsity Marketing Models

This repository provides a PyTensor-based implementation of a batched, tempered Sequential Monte Carlo (SMC) framework specifically designed for semi-continuous Hidden Markov Models (HMMs) and high-sparsity transaction data (85-98% zeros).

## Key Features
- **GPU Acceleration:** Batched forward filtering for order-of-magnitude speedups over serial estimation.
- **HMC Failure Diagnostics:** Automated reporting of NUTS/HMC failure modes (R-hat, ESS, Divergences) in sparse regimes.
- **Flexible Specifications:** Support for NBD, Hurdle, Gamma, and HMM specifications.

## The "Computational Constraint" Finding
Our research demonstrates that marketing researchers often favor simple models not due to lack of theoretical interest in dynamic specifications, but because complex models were previously inestimable in high-sparsity environments. This library removes that barrier.

## Simulation: The 4 Worlds
We validate parameter recovery across four transactional regimes:
1. **Poisson World:** Low sparsity, low skew.
2. **Standard RFM:** Medium sparsity.
3. **Clumpy World:** High sparsity, high skew.
4. **The Cliff:** Extreme sparsity (95%+ zeros).

## Installation
```bash
pip install -r requirements.txt
# Requires: pymc >= 5.0, pytensor, numpy, pandas
