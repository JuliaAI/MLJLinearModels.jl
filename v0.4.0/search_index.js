var documenterSearchIndex = {"docs":
[{"location":"api/#API-1","page":"API","title":"API","text":"","category":"section"},{"location":"api/#Standalone-1","page":"API","title":"Standalone","text":"","category":"section"},{"location":"api/#Regression-1","page":"API","title":"Regression","text":"","category":"section"},{"location":"api/#","page":"API","title":"API","text":"Standard constructors","category":"page"},{"location":"api/#","page":"API","title":"API","text":"LinearRegression\nRidgeRegression\nLassoRegression\nElasticNetRegression\nHuberRegression\nQuantileRegression\nLADRegression","category":"page"},{"location":"api/#MLJLinearModels.LinearRegression","page":"API","title":"MLJLinearModels.LinearRegression","text":"LinearRegression(; fit_intercept)\n\n\nObjective function: Xθ - y₂²2.\n\n\n\n\n\n","category":"function"},{"location":"api/#MLJLinearModels.RidgeRegression","page":"API","title":"MLJLinearModels.RidgeRegression","text":"RidgeRegression()\nRidgeRegression(λ; lambda, fit_intercept, penalize_intercept)\n\n\nObjective function: Xθ - y₂²2 + λθ₂²2.\n\n\n\n\n\n","category":"function"},{"location":"api/#MLJLinearModels.LassoRegression","page":"API","title":"MLJLinearModels.LassoRegression","text":"LassoRegression()\nLassoRegression(λ; lambda, fit_intercept, penalize_intercept)\n\n\nObjective function: Xθ - y₂²2 + λθ₁.\n\n\n\n\n\n","category":"function"},{"location":"api/#MLJLinearModels.ElasticNetRegression","page":"API","title":"MLJLinearModels.ElasticNetRegression","text":"ElasticNetRegression()\nElasticNetRegression(λ)\nElasticNetRegression(λ, γ; lambda, gamma, fit_intercept, penalize_intercept)\n\n\nObjective function: Xθ - y₂²2 + λθ₂²2 + γθ₁.\n\n\n\n\n\n","category":"function"},{"location":"api/#MLJLinearModels.HuberRegression","page":"API","title":"MLJLinearModels.HuberRegression","text":"HuberRegression()\nHuberRegression(δ)\nHuberRegression(δ, λ)\nHuberRegression(δ, λ, γ; delta, lambda, gamma, penalty, fit_intercept, penalize_intercept)\n\n\nHuber Regression with objective:\n\nρ(Xθ - y) + λθ₂²2 + γθ₁\n\nWhere ρ is the Huber function ρ(r) = r²/2if|r|≤δandρ(r)=δ(|r|-δ/2)` otherwise.\n\n\n\n\n\n","category":"function"},{"location":"api/#MLJLinearModels.QuantileRegression","page":"API","title":"MLJLinearModels.QuantileRegression","text":"QuantileRegression()\nQuantileRegression(δ)\nQuantileRegression(δ, λ)\nQuantileRegression(δ, λ, γ; delta, lambda, gamma, penalty, fit_intercept, penalize_intercept)\n\n\nQuantile Regression with objective:\n\nρ(Xθ - y) + λθ₂²2 + γθ₁\n\nWhere ρ is the check function ρ(r) = r(δ - 1(r < 0)).\n\n\n\n\n\n","category":"function"},{"location":"api/#MLJLinearModels.LADRegression","page":"API","title":"MLJLinearModels.LADRegression","text":"LADRegression()\nLADRegression(λ)\nLADRegression(λ, γ; lambda, gamma, penalty, fit_intercept, penalize_intercept)\n\n\nLeast Absolute Deviation regression with objective:\n\nXθ - y₁ + λθ₂²2 + γθ₁\n\nThis is a specific type of Quantile Regression with δ=0.5 (median).\n\n\n\n\n\n","category":"function"},{"location":"api/#","page":"API","title":"API","text":"Generic constructors","category":"page"},{"location":"api/#","page":"API","title":"API","text":"GeneralizedLinearRegression\nRobustRegression","category":"page"},{"location":"api/#MLJLinearModels.GeneralizedLinearRegression","page":"API","title":"MLJLinearModels.GeneralizedLinearRegression","text":"GeneralizedLinearRegression{L<:Loss, P<:Penalty}\n\nGeneralized Linear Regression (GLR) model with objective function:\n\nL(y Xθ) + P(θ)\n\nwhere L is a loss function, P a penalty, y is the vector of observed response, X is the feature matrix and θ the vector of parameters.\n\nSpecial cases include:\n\nOLS regression:      L2 loss, no penalty.\nRidge regression:    L2 loss, L2 penalty.\nLasso regression:    L2 loss, L1 penalty.\nLogistic regression: Logit loss, [no,L1,L2] penalty.\n\n\n\n\n\n","category":"type"},{"location":"api/#MLJLinearModels.RobustRegression","page":"API","title":"MLJLinearModels.RobustRegression","text":"RobustRegression()\nRobustRegression(ρ)\nRobustRegression(ρ, λ)\nRobustRegression(ρ, λ, γ; rho, lambda, gamma, penalty, fit_intercept, penalize_intercept)\n\n\nObjective function: ρ(Xθ - y) + λθ₂² + γθ₁ where ρ is a given function on the residuals.\n\n\n\n\n\n","category":"function"},{"location":"api/#Classification-1","page":"API","title":"Classification","text":"","category":"section"},{"location":"api/#","page":"API","title":"API","text":"LogisticRegression\nMultinomialRegression","category":"page"},{"location":"api/#MLJLinearModels.LogisticRegression","page":"API","title":"MLJLinearModels.LogisticRegression","text":"LogisticRegression()\nLogisticRegression(λ)\nLogisticRegression(λ, γ; lambda, gamma, penalty, fit_intercept, penalize_intercept, multi_class, nclasses)\n\n\nObjective function: L(y Xθ) + λθ₂²2 + γθ₁ where L is either the logistic loss in the binary case or the multinomial loss otherwise.\n\n\n\n\n\n","category":"function"},{"location":"api/#MLJLinearModels.MultinomialRegression","page":"API","title":"MLJLinearModels.MultinomialRegression","text":"MultinomialRegression(a; kwa...)\n\n\nObjective function: L(y Xθ) + λθ₂²2 + γθ₁ where L is the multinomial loss.\n\n\n\n\n\n","category":"function"},{"location":"api/#MLJ-Interface-1","page":"API","title":"MLJ Interface","text":"","category":"section"},{"location":"api/#Regressors-1","page":"API","title":"Regressors","text":"","category":"section"},{"location":"api/#","page":"API","title":"API","text":"LinearRegressor\nRidgeRegressor\nLassoRegressor\nElasticNetRegressor\nHuberRegressor\nQuantileRegressor\nLADRegressor\nRobustRegressor","category":"page"},{"location":"api/#MLJLinearModels.LinearRegressor","page":"API","title":"MLJLinearModels.LinearRegressor","text":"Standard linear regression model.\n\nParameters\n\nfit_intercept (Bool): whether to fit the intercept or not.\nsolver: type of solver to use (if nothing the default is used). The           solver is Cholesky by default but can be Conjugate-Gradient as           well. See ?Analytical for more information.\n\n\n\n\n\n","category":"type"},{"location":"api/#MLJLinearModels.RidgeRegressor","page":"API","title":"MLJLinearModels.RidgeRegressor","text":"Ridge regression model with objective function\n\nXθ - y₂²2 + λθ₂²2\n\nParameters\n\nlambda (Real): strength of the L2 regularisation.\nfit_intercept (Bool): whether to fit the intercept or not.\npenalize_intercept (Bool): whether to penalize the intercept.\nsolver: type of solver to use (if nothing the default is used). The           solver is Cholesky by default but can be Conjugate-Gradient as           well. See ?Analytical for more information.\n\n\n\n\n\n","category":"type"},{"location":"api/#MLJLinearModels.LassoRegressor","page":"API","title":"MLJLinearModels.LassoRegressor","text":"Lasso regression model with objective function\n\nXθ - y₂²2 + λθ₁\n\nParameters\n\nlambda (Real): strength of the L1 regularisation.\nfit_intercept (Bool): whether to fit the intercept or not.\npenalize_intercept (Bool): whether to penalize the intercept.\nsolver: type of solver to use (if nothing the default is used). Either           FISTA or ISTA can be used (proximal methods, with/without           acceleration).\n\n\n\n\n\n","category":"type"},{"location":"api/#MLJLinearModels.ElasticNetRegressor","page":"API","title":"MLJLinearModels.ElasticNetRegressor","text":"Elastic net regression model with objective function\n\nXθ - y₂²2 + λθ₂²2 + γθ₁\n\nParameters\n\nlambda (Real): strength of the L2 regularisation.\ngamma (Real): strength of the L1 regularisation.\nfit_intercept (Bool): whether to fit the intercept or not.\npenalize_intercept (Bool): whether to penalize the intercept.\nsolver: type of solver to use (if nothing the default is used). Either           FISTA or ISTA can be used (proximal methods, with/without           acceleration).\n\n\n\n\n\n","category":"type"},{"location":"api/#MLJLinearModels.HuberRegressor","page":"API","title":"MLJLinearModels.HuberRegressor","text":"Huber Regression, see RobustRegressor, it's the same but with the robust loss set to HuberRho.  The parameters are the same apart from delta which parametrises the HuberRho function (radius of the ball within which the loss is a quadratic loss).\n\n\n\n\n\n","category":"type"},{"location":"api/#MLJLinearModels.QuantileRegressor","page":"API","title":"MLJLinearModels.QuantileRegressor","text":"Quantile Regression, see RobustRegressor, it's the same but with the robust loss set to QuantileRho.  The parameters are the same apart from delta which parametrises the QuantileRho function (indicating the  quantile to use with default 0.5 for the median regression).\n\n\n\n\n\n","category":"type"},{"location":"api/#MLJLinearModels.LADRegressor","page":"API","title":"MLJLinearModels.LADRegressor","text":"Least Absolute Deviation regression with with objective function\n\nρ(Xθ - y) + λθ₂² + γθ₁\n\nwhere ρ is the absolute loss.\n\nSee also RobustRegressor.\n\n\n\n\n\n","category":"type"},{"location":"api/#MLJLinearModels.RobustRegressor","page":"API","title":"MLJLinearModels.RobustRegressor","text":"Robust regression model with objective function\n\nρ(Xθ - y) + λθ₂² + γθ₁\n\nwhere ρ is a robust loss function (e.g. the Huber function).\n\nParameters\n\nrho (RobustRho): the type of robust loss to use (see HuberRho,                    TalwarRho, ...)\npenalty (Symbol or String): the penalty to use, either :l2, :l1, :en                               (elastic net) or :none. (Default: :l2)\nlambda (Real): strength of the regulariser if penalty is :l2 or :l1.                  Strength of the L2 regulariser if penalty is :en.\ngamma (Real): strength of the L1 regulariser if penalty is :en.\nfit_intercept (Bool): whether to fit an intercept (Default: true)\npenalize_intercept (Bool): whether to penalize intercept (Default: false)\nsolver (Solver): type of solver to use, default if nothing.\n\n\n\n\n\n","category":"type"},{"location":"api/#Classifiers-1","page":"API","title":"Classifiers","text":"","category":"section"},{"location":"api/#","page":"API","title":"API","text":"LogisticClassifier\nMultinomialClassifier","category":"page"},{"location":"api/#MLJLinearModels.LogisticClassifier","page":"API","title":"MLJLinearModels.LogisticClassifier","text":"Logistic Classifier (typically called \"Logistic Regression\"). This model is a standard classifier for both binary and multiclass classification. In the binary case it corresponds to the LogisticLoss, in the multiclass to the Multinomial (softmax) loss. An elastic net penalty can be applied with overall objective function\n\nL(y Xθ) + λθ₂²2 + γθ₁\n\nWhere L is either the logistic or multinomial loss and λ and γ indicate the strength of the L2 (resp. L1) regularisation components.\n\nParameters\n\npenalty (Symbol or String): the penalty to use, either :l2, :l1, :en                               (elastic net) or :none. (Default: :l2)\nlambda (Real): strength of the regulariser if penalty is :l2 or :l1.                  Strength of the L2 regulariser if penalty is :en.\ngamma (Real): strength of the L1 regulariser if penalty is :en.\nfit_intercept (Bool): whether to fit an intercept (Default: true)\npenalize_intercept (Bool): whether to penalize intercept (Default: false)\nsolver (Solver): type of solver to use, default if nothing.\nmulti_class (Bool): whether it's a binary or multi class classification                       problem. This is usually set automatically.\n\n\n\n\n\n","category":"type"},{"location":"api/#MLJLinearModels.MultinomialClassifier","page":"API","title":"MLJLinearModels.MultinomialClassifier","text":"See LogisticClassifier, it's the same except that multi_class is set to true by default. The other parameters are the same.\n\n\n\n\n\n","category":"type"},{"location":"quickstart/#Quick-start-1","page":"Quick start","title":"Quick start","text":"","category":"section"},{"location":"quickstart/#Using-MLJLinearModels-by-itself-1","page":"Quick start","title":"Using MLJLinearModels by itself","text":"","category":"section"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"The package works by","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"specifying the kind of model you want along with its hyper-parameters,\ncalling fit with that model and the data: fit(model, X, y).","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"note: Note\nThe convention in this  package is that the feature matrix has dimensions n times p where n is the number of records (points) and p is the number of features (dimensions).","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"Below we show an example of regression and an example of classification.","category":"page"},{"location":"quickstart/#Regression-1","page":"Quick start","title":"Regression","text":"","category":"section"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"The lasso regression corresponds to a l2-loss function with a l1-penalty:","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"theta_textLasso = frac12y-Xtheta_2^2 + lambdatheta_1","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"which you can create as follows:","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"n = 500\np = 5\nX = randn(n, p)\ny = randn(n)\nλ = 0.7\nlasso = LassoRegression(λ)\ntheta = fit(lasso, X, y)","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"By default this fits an intercept so that the dimension of theta in the example above is p+1, the last element being the intercept.","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"So if you wanted to compute the RMSE norm of the residuals you would do","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"r = y - hcat(X, ones(n)) * theta\ne = sqrt(sum(abs2.(r)) / n)","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"You can also just compute the objective:","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"o = objective(lasso, X, y) # function of theta\no(theta) # value at the theta obtained from the fit","category":"page"},{"location":"quickstart/#Classification-1","page":"Quick start","title":"Classification","text":"","category":"section"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"note: Note\nThe convention in this  package for binary classification is that the entries of y are pm 1 while for multiclass classification the entries of y are 1dotsc where c is the number of classes. If you use MLJ you won't have to  think about this.","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"Here's an example for a logistic classifier (binary classification) with a standard L2 regularisation:","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"n = 500\np = 5\nX = randn(n, p)\ny = 2 * (rand(n) .< 0.5) .- 1   # entries are +-1\nλ = 0.5\nlogistic = LogisticRegression(λ)\ntheta = fit(logistic, X, y)","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"The process for a multiclass classification is identical (you can either call LogisticRegression or MultinomialRegression it will lead to the same model). The  only difference is that the encoding of the target is  expected to  be {1, ..., c} where c is the number of classes.","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"Note that for a multiclass classification, theta is a vector of dimension p times c or (p+1)times c depending on whether an intercept is fitted or not. To make sense of that vector you can reshape it as follows (assuming no intercept is fitted):","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"W = reshape(theta, p, c)","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"where W is a matrix with each column corresponding to each of the c classes. If you needed to predict using that matrix you would do XW which would give you a matrix of size n times p on which you could apply a softmax for each row to get a score per class for each instance (i.e. a normalised matrix of size ntimes p where you can interpret the entry (ij) as the score attributed by the model to example i to belong in class j).","category":"page"},{"location":"quickstart/#Using-MLJLinearModels-with-MLJ-1","page":"Quick start","title":"Using MLJLinearModels with MLJ","text":"","category":"section"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"Using MLJLinearModels in the context of MLJ allows to benefit from tools for encoding data, dealing with missing values, keeping track of class labels, doing hyper-parameter tuning, composing models, etc.","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"In order to load a model from MLJLinearModels you need to call @load model_name pkg=MLJLinearModels where model_name follows the MLJ conventions and is one of","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"(Regression): LinearRegressor, RidgeRegressor, LassoRegressor, ElasticNetRegressor, RobustRegressor, HuberRegressor, QuantileRegressor, LADRegressor\n(Classification): LogisticClassifier, MultinomialClassifier","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"Note that the names are slightly different (ending in Regressor or Classifier).","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"Check out the MLJ documentation or at the MLJ Tutorials for more information on MLJ itself.","category":"page"},{"location":"quickstart/#Regression-2","page":"Quick start","title":"Regression","text":"","category":"section"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"Let's fit a simple Huber regression on the boston dataset.","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"using MLJ\n@load HuberRegressor pkg=MLJLinearModels\n\nX, y = @load_boston\nmdl = HuberRegressor()\nmach = machine(mdl, X, y)\nfit!(mach)\nparams = fitted_params(mach)\n\nparams.coefs # coefficient of the regression with names\nparams.intercept # intercept","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"MLJ makes it seamless to do  prediction as well:","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"ypred = predict(mach, X)","category":"page"},{"location":"quickstart/#Classification-2","page":"Quick start","title":"Classification","text":"","category":"section"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"Let's fit a simple multiclass classifier on the Iris dataset","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"using MLJ\n@load MultinomialClassifier pkg=MLJLinearModels\n\nX, y = @load_iris\nmdl = MultinomialClassifier(lambda=0.5, gamma=0.7)\nmach = machine(mdl, X, y)\nfit!(mach)\nparams = fitted_params(mach)\n\nparams.coefs # coefficients of the regression\nparams.intercept # intercepts","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"Note: for a multiclass classification like the one above, each class gets its own model so for instance params.intercept has 3 values, likewise params.coefs.sepal_length has 3 values.","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"Predictions are easy too, note that this is a probabilistic model: it returns scores per class:","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"ypred = predict(mach, X)\nypred[1]","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"That first element is a UnivariateFinite distribution object which keeps track of each class labels (setosa, versicolor, virginica) and a score for each class (in my case: 0.991, 0.009 and 0).","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"You can collapse that to a single prediction if you would like using  predict_mode:","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"ypred = predict_mode(mach, rows=1:2)","category":"page"},{"location":"quickstart/#","page":"Quick start","title":"Quick start","text":"Which, in my case, gives setosa, setosa (correct in both cases).","category":"page"},{"location":"models/#Available-models-1","page":"Models","title":"Available models","text":"","category":"section"},{"location":"models/#","page":"Models","title":"Models","text":"Your help is welcome to help extend these lists. Note the current limitations:","category":"page"},{"location":"models/#","page":"Models","title":"Models","text":"The models are built and tested assuming n > p; if this doesn't hold, tricks should be employed to speed up computations; these have not been implemented yet.\nCV-aware code not implemented yet (code that re-uses computations when fitting over a number of hyper-parameters);  \"Meta\" functionalities such as One-vs-All or Cross-Validation are left to other packages such as MLJ.\nNo support yet for sparse matrices.\nStochastic solvers have not yet been implemented.\nAll computations are assumed to be done in Float64.","category":"page"},{"location":"models/#Regression-models-1","page":"Models","title":"Regression models","text":"","category":"section"},{"location":"models/#","page":"Models","title":"Models","text":"Regressors Formulation¹ Available solvers Comments\nOLS & Ridge L2Loss + 0/L2 Analytical² or CG³ \nLasso & Elastic-Net L2Loss + 0/L2 + L1 (F)ISTA⁴ \nRobust 0/L2 RobustLoss⁵ + 0/L2 Newton, NewtonCG, LBFGS, IWLS-CG⁶ no scale⁷\nRobust L1/EN RobustLoss + 0/L2 + L1 (F)ISTA \nQuantile⁸ + 0/L2 RobustLoss + 0/L2 LBFGS, IWLS-CG \nQuantile L1/EN RobustLoss + 0/L2 + L1 (F)ISTA ","category":"page"},{"location":"models/#","page":"Models","title":"Models","text":"\"0\" stands for no penalty\nAnalytical means the solution is computed in \"one shot\" using the \\ solver,\nCG = conjugate gradient\n(Accelerated) Proximal Gradient Descent\nHuber, Andrews, Bisquare, Logistic, Fair and Talwar weighing functions available.\nIteratively re-Weighted Least Squares where each system is solved iteratively via CG\nIn other packages such as Scikit-Learn, a scale factor is estimated along with the parameters, this is a bit ad-hoc and corresponds more to a statistical perspective, further it does not work well with penalties; we recommend using cross-validation to set the parameter of the Huber Loss.\nIncludes as special case the least absolute deviation (LAD) regression when δ=0.5.","category":"page"},{"location":"models/#Classification-models-1","page":"Models","title":"Classification models","text":"","category":"section"},{"location":"models/#","page":"Models","title":"Models","text":"Classifiers Formulation Available solvers Comments\nLogistic 0/L2 LogisticLoss + 0/L2 Newton, Newton-CG, LBFGS yᵢ∈{±1}\nLogistic L1/EN LogisticLoss + 0/L2 + L1 (F)ISTA yᵢ∈{±1}\nMultinomial 0/L2 MultinomialLoss + 0/L2 Newton-CG, LBFGS yᵢ∈{1,...,c}\nMultinomial L1/EN MultinomialLoss + 0/L2 + L1 ISTA, FISTA yᵢ∈{1,...,c}","category":"page"},{"location":"models/#","page":"Models","title":"Models","text":"Unless otherwise specified:","category":"page"},{"location":"models/#","page":"Models","title":"Models","text":"Newton-like solvers use Hager-Zhang line search (default in Optim.jl)\nISTA, FISTA solvers use backtracking line search and a shrinkage factor of β=0.8","category":"page"},{"location":"models/#","page":"Models","title":"Models","text":"Note: these models were all tested for correctness whenever a direct comparison with another package was possible, usually by comparing the objective function at the coefficients returned (cf. the tests):","category":"page"},{"location":"models/#","page":"Models","title":"Models","text":"(against scikit-learn): Lasso, Elastic-Net, Logistic (L1/L2/EN), Multinomial (L1/L2/EN)\n(against quantreg): Quantile (0/L1)","category":"page"},{"location":"models/#","page":"Models","title":"Models","text":"Systematic timing benchmarks have not been run yet but it's planned (see this issue).","category":"page"},{"location":"solvers/#Solvers-1","page":"Solvers","title":"Solvers","text":"","category":"section"},{"location":"solvers/#","page":"Solvers","title":"Solvers","text":"In general MLJLinearModels tries to use \"reasonable defaults\" for solvers. You may want to pick something else particularly if your data is extreme in some way (e.g. very noisy, or very large).","category":"page"},{"location":"solvers/#","page":"Solvers","title":"Solvers","text":"Only some solvers are appropriate for some models (see models) for a list.","category":"page"},{"location":"solvers/#","page":"Solvers","title":"Solvers","text":"Analytical\nNewton\nNewtonCG\nLBFGS\nProxGrad\nIWLSCG","category":"page"},{"location":"solvers/#MLJLinearModels.Analytical","page":"Solvers","title":"MLJLinearModels.Analytical","text":"Analytical solver (Cholesky). If the iterative parameter is set to true then a CG solver is used. The CG solver is matrix-free and should be preferred in \"large scale\" cases (when the hat matrix X'X is \"big\").\n\nParameters\n\niterative (Bool): whether to use CG (iterative) or not\nmax_inner (Int): in the iterative mode, how many inner iterations to do.\n\n\n\n\n\n","category":"type"},{"location":"solvers/#MLJLinearModels.Newton","page":"Solvers","title":"MLJLinearModels.Newton","text":"Newton solver. This is a full Hessian solver and should be avoided for \"large scale\" cases.\n\n\n\n\n\n","category":"type"},{"location":"solvers/#MLJLinearModels.NewtonCG","page":"Solvers","title":"MLJLinearModels.NewtonCG","text":"Newton CG solver. This is the same as the Newton solver except that instead of solving systems of the form H\\b where H is the full Hessian, it uses a matrix-free conjugate gradient approach to solving that system. This should generally be preferred for larger scale cases.\n\n\n\n\n\n","category":"type"},{"location":"solvers/#MLJLinearModels.LBFGS","page":"Solvers","title":"MLJLinearModels.LBFGS","text":"LBFGS quasi-Newton solver. See the wikipedia entry.\n\n\n\n\n\n","category":"type"},{"location":"solvers/#MLJLinearModels.ProxGrad","page":"Solvers","title":"MLJLinearModels.ProxGrad","text":"Proximal Gradient solver for non-smooth objective functions.\n\nParameters\n\naccel (Bool): whether to use Nesterov-style acceleration\nmax_iter (Int): number of overall iterations\ntol (Float64): tolerance for the relative change θ ie norm(θ-θ_)/norm(θ)\nmax_inner: number of inner steps when searching for a stepsize in the              backtracking step\nbeta: rate of shrinkage in the backtracking step (between 0 and 1)\n\n\n\n\n\n","category":"type"},{"location":"solvers/#MLJLinearModels.IWLSCG","page":"Solvers","title":"MLJLinearModels.IWLSCG","text":"Iteratively Reweighted Least Square with Conjugate Gradient. This is the standard (expensive) IWLS but with more efficient solves to avoid full matrix computations.\n\nParameters\n\nmax_iter (Int): number of max iterations (outer)\nmax_inner (Int): number of iterations for the CG solves\ntol (Float64): tolerance for the relative change θ ie norm(θ-θ_)/norm(θ)\ndamping (Float64): how much to trust iterates (1=full trust)\nthreshold (Float64): threshold for the residuals\n\n\n\n\n\n","category":"type"},{"location":"#MLJLinearModels.jl-1","page":"Home","title":"MLJLinearModels.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"This is a convenience package gathering functionalities to solve a number of generalised linear regression/classification problems which, inherently, correspond to an optimisation problem of the form","category":"page"},{"location":"#","page":"Home","title":"Home","text":"L(y Xtheta) + P(theta)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"where:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"y is the target or response, a vector of length n either of real values (regression) or integers (classification),\nX is the design or feature matrix, a matrix of real values of size n times p where p is the number of features or dimensions,\n\ntheta is a vector of p real valued coefficients to determine,\nL is a loss function, a pre-determined function of mathbb R^n to mathbb R^+ penalising the amplitude of the residuals in a specific way,\nP is a penalty function, a pre-determined function of mathbb R^n to mathbb R^+ penalising the amplitude of the  coefficients in a specific way.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"A well known example is the Ridge regression where the objective is to minimise:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"y - Xtheta_2^2 + lambdatheta_2^2","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Head to the Quick Start page to get an idea of how this package works.","category":"page"},{"location":"#What-this-package-aims-to-do-1","page":"Home","title":"What this package aims to do","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"make these regressions models \"easy to call\" and callable in a unified way,\nseamless interface with MLJ.jl,\nfocus on performance including in \"big data\" settings exploiting packages such as Optim.jl, and IterativeSolvers.jl,","category":"page"},{"location":"#","page":"Home","title":"Home","text":"All models allow to fit an intercept and allow the penalty to be optionally applied on the intercept. All models attempt to be efficient in terms of memory allocation to avoid unnecessary copies of the data.","category":"page"},{"location":"#What-this-package-does-not-aim-to-do-1","page":"Home","title":"What this package does not aim to do","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"This package deliberately does not offer the following features","category":"page"},{"location":"#","page":"Home","title":"Home","text":"facilities for data pre-processing (use MLJ for that),\nfacilities for hyperparameter tuning (use MLJ for that)\n\"significance\" statistics (consider GLM for that)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The last point is important, the package assumes that the user has some principled way of picking an appropriate loss function / penalty. The package makes no assumption of normality etc which befalls the realm of statistics.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"You can still build data-driven uncertainty estimates around your parameters if you so desire by using Bootstrap.jl.","category":"page"}]
}
