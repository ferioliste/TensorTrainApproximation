# Randomized algorithms for tensor train approximation

The tensor train format (TT-format) allows the representation of a tensor through the contraction of several three-dimensional tensors called TT-cores. The purpose of this project is to explore how randomization can be used to achieve performance improvements in TT-factorization and TT-rounding algorithms. This repository contains an implementation of the analyzed algorithms.

The code was developed for the semestral project "Randomized algorithms for tensor train approximation" at EPFL in the academic year 2023-2024, spring semester.

Check out the final report here: [`Stefano_FERIOLI_semester_project.pdf`](./Stefano_FERIOLI_semester_project.pdf).

Final grade: 5.75/6

## Implemented algorithms
When installed, the package enables 7 new methods:

- `tt_svd`: the classical TT-factorization algorithm presented in [[Oseledets, 2011]](https://www.researchgate.net/profile/Ivan-Oseledets/publication/220412263_Tensor-Train_Decomposition/links/5bbfb5c5299bf1004c5a56e3/Tensor-Train-Decomposition.pdf).
- `tt_svd_rounding`: the classical TT-rounding algorithm presented in [[Oseledets, 2011]](https://www.researchgate.net/profile/Ivan-Oseledets/publication/220412263_Tensor-Train_Decomposition/links/5bbfb5c5299bf1004c5a56e3/Tensor-Train-Decomposition.pdf).

- `tt_rsvd`: the TT-factorization algorithm obtained from `tt_svd` by replacing the truncated SVDs with a basic version of the randomized range finder.
- `tt_rsvd_rounding`: the TT-rounding algorithm obtained from `tt_svd_rounding` by replacing the truncated SVDs with a basic version of the randomized range finder. The algorithm is also presented in [[Al Daas et al., 2021]](https://arxiv.org/pdf/2110.04393).
- `tt_rand_orth_rounding`: the TT-rounding algorithm  Randomize-then-Orthogonalize presented in [[Al Daas et al., 2021]](https://arxiv.org/pdf/2110.04393).

- `tt_STTA`: the TT-factorization algorithm presented in [[Kressner et al., 2022]](https://arxiv.org/pdf/2208.02600).
- `tt_STTA_rounding`: the TT-rounding algorithm presented in [[Kressner et al., 2022]](https://arxiv.org/pdf/2208.02600).

Note that all randomized methods allow to choose the `sketch_type` among `gaussian`, `srht`, and `srht_hash`. The random seed can also be set for reproducibility.
