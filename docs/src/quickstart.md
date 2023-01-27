# Quick start

## Using MLJLinearModels by itself

The package works by

1. specifying the kind of model you want along with its hyper-parameters,
2. calling `fit` with that model and the data: `fit(model, X, y)`.

!!! note

    The convention in this  package is that the feature matrix has dimensions ``n \times p`` where ``n`` is the number of records (points) and ``p`` is the number of features (dimensions).

Below we show an example of regression and an example of classification.

### Regression

The lasso regression corresponds to a l2-loss function with a l1-penalty:

```math
\theta_{\text{Lasso}} = \frac12\|y-X\theta\|_2^2 + \lambda\|\theta\|_1
```

which you can create as follows:

```julia
n = 500
p = 5
X = randn(n, p)
y = randn(n)
λ = 0.7
lasso = LassoRegression(λ)
theta = fit(lasso, X, y)
```

By default this fits an intercept so that the dimension of `theta` in the example above is `p+1`, the **last** element being the intercept.

So if you wanted to compute the RMSE norm of the residuals you would do

```julia
r = y - hcat(X, ones(n)) * theta
e = sqrt(sum(abs2.(r)) / n)
```

You can also just compute the objective:

```julia
o = objective(lasso, X, y) # function of theta
o(theta) # value at the theta obtained from the fit
```

### Classification

!!! note

    The convention in this  package for **binary** classification is that the entries of ``y`` are ``\{\pm 1\}`` while for **multiclass** classification the entries of ``y`` are ``\{1,\dots,c\}`` where `c` is the number of classes. If you use MLJ you won't have to  think about this.

Here's an example for a logistic classifier (binary classification) with a standard L2 regularisation:

```julia
n = 500
p = 5
X = randn(n, p)
y = 2 * (rand(n) .< 0.5) .- 1   # entries are +-1
λ = 0.5
logistic = LogisticRegression(λ)
theta = fit(logistic, X, y)
```

The process for a multiclass classification is identical (you can either call `LogisticRegression` or `MultinomialRegression` it will lead to the same model). The  only difference is that the encoding of the target is  expected to  be `{1, ..., c}` where `c` is the number of classes.

Note that for a multiclass classification, `theta` is a **vector** of dimension ``p \times c`` or ``(p+1)\times c`` depending on whether an intercept is fitted or not. To make sense of that vector you can _reshape_ it as follows (assuming no intercept is fitted):

```julia
W = reshape(theta, p, c)
```

where `W` is a matrix with each column corresponding to each of the `c` classes. If you needed to predict using that matrix you would do ``XW`` which would give you a matrix of size ``n \times p`` on which you could apply a softmax for each row to get a score per class for each instance (i.e. a normalised matrix of size ``n\times p`` where you can interpret the entry ``(i,j)`` as the score attributed by the model to example `i` to belong in class `j`).

## Using MLJLinearModels with MLJ

Using MLJLinearModels in the context of MLJ allows to benefit from tools for encoding data, dealing with missing values, keeping track of class labels, doing hyper-parameter tuning, composing models, etc.

In order to load a model from MLJLinearModels you need to call `@load model_name pkg=MLJLinearModels` where `model_name` follows the MLJ conventions and is one of

* (Regression): `LinearRegressor`, `RidgeRegressor`, `LassoRegressor`, `ElasticNetRegressor`, `RobustRegressor`, `HuberRegressor`, `QuantileRegressor`, `LADRegressor`
* (Classification): `LogisticClassifier`, `MultinomialClassifier`

Note that the names are slightly different (ending in _Regressor_ or _Classifier_).

Check out the [MLJ documentation](https://alan-turing-institute.github.io/MLJ.jl/stable/) or at the [MLJ Tutorials](https://alan-turing-institute.github.io/MLJTutorials/) for more information on MLJ itself.

### Regression

Let's fit a simple [Huber regression](https://en.wikipedia.org/wiki/Huber_loss) on the [boston](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) dataset.

```julia
using MLJ
@load HuberRegressor pkg=MLJLinearModels

X, y = @load_boston
mdl = HuberRegressor()
mach = machine(mdl, X, y)
fit!(mach)
params = fitted_params(mach)

params.coefs # coefficient of the regression with names
params.intercept # intercept
```

MLJ makes it seamless to do  prediction as well:

```julia
ypred = predict(mach, X)
```

### Classification

Let's fit a simple multiclass classifier on the Iris dataset

```julia
using MLJ
@load MultinomialClassifier pkg=MLJLinearModels

X, y = @load_iris
mdl = MultinomialClassifier(lambda=0.5, gamma=0.7)
mach = machine(mdl, X, y)
fit!(mach)
params = fitted_params(mach)

params.coefs # coefficients of the regression
params.intercept # intercepts
```

**Note**: for a multiclass classification like the one above, each class gets its own model so for instance `params.intercept` has 3 values, likewise `params.coefs.sepal_length` has 3 values.

Predictions are easy too, note that this is a _probabilistic model_: it returns **scores** per class:

```julia
ypred = predict(mach, X)
ypred[1]
```

That first element is a `UnivariateFinite` distribution object which keeps track of each class labels (`setosa`, `versicolor`, `virginica`) and a score for each class (in my case: `0.991`, `0.009` and `0`).

You can collapse that to a single prediction if you would like using  `predict_mode`:

```julia
ypred = predict_mode(mach, rows=1:2)
```

Which, in my case, gives `setosa`, `setosa` (correct in both cases).

### Customizing the solvers

Depending on your problem you way want to customize the default solver or use a diffrent one. Since this package uses [Optim](https://julianlsolvers.github.io/Optim.jl/stable/) behind the scene, we can interact directly with this package.

For instance, we may want to be more stringent about the convergence criterion of the LBFGS solver. This can be done by changing the general Optim `f_tol` parameter which defaults to ``10^{-4}``:

```julia
import Optim

new_optim_option = Optim.Options(f_tol=0)
mdl = MultinomialClassifier(solver=LBFGS(optim_options=new_optim_option))
mach = machine(mdl, X, y)
fit!(mach)
```

For a full description of available solvers and API, see: [Solvers](@ref).
