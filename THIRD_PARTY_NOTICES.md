# Third-Party Notices

This repository contains modified scripts derived from code in:

- Renata Wong, `quantum-bluestein`
- Original repository: https://github.com/renatawong/quantum-bluestein
- License: MIT License
- Original author/copyright holder: Renata Wong

The following files are modified or extended from Renata Wong's original code:

1. `scripts/qba_mixed_radix_aqft_unified_benchmark_v4_largeN.py`

   Derived in part from:
   - `standard-mixed-radix-qft.ipynb`
   - Original repository: https://github.com/renatawong/quantum-bluestein

   Modifications include:
   - unified QBA / mixed-radix QFT / AQFT benchmark pipeline;
   - common Qiskit transpilation settings;
   - factorization metadata;
   - gate-count, two-qubit-gate, depth, and fidelity outputs;
   - paper-facing plots and LaTeX table generation.

2. `scripts/quantum-bluestein_revised_v5_paper_aligned_N3_worked_example.py`

   Derived in part from:
   - `quantum-bluestein-n3.ipynb`
   - Original repository: https://github.com/renatawong/quantum-bluestein

   Modifications include:
   - pure Python execution support;
   - automatic figure generation;
   - paper-aligned logical histogram;
   - worked-example input state aligned with Appendix A.

3. `scripts/quantum-bluestein_revised_v5_paper_aligned_N6_worked_example.py`

   Derived in part from:
   - `quantum-bluestein-n6.ipynb`
   - Original repository: https://github.com/renatawong/quantum-bluestein

   Modifications include:
   - pure Python execution support;
   - automatic circuit and histogram figure export;
   - paper-aligned \(N=6\) worked-example input;
   - post-selected logical-output analysis.

Unless otherwise stated, the modified scripts are distributed under the MIT License,
with attribution to both the original author and the modifier.