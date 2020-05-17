# Solvers

In general MLJLinearModels tries to use "reasonable defaults" for solvers. You may want to pick something else particularly if your data is extreme in some way (e.g. very noisy, or very large).

Only some solvers are appropriate for some models (see [models](/models/)) for a list.

```@docs
Analytical
Newton
NewtonCG
LBFGS
ProxGrad
IWLSCG
```
