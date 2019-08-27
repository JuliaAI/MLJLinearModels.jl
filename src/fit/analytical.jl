# Solvers corresponding to solving a closed-form expression

"""
$SIGNATURES

Fit a least square regression either with no penalty (OLS) or with a L2 penalty (Ridge).

## Complexity

Assuming `n` dominates `p`,

* non-iterative (full solve):     O(np²) - dominated by the construction of the Hessian X'X.
* iterative (conjugate gradient): O(κnp) - with κ the number of CG steps (κ ≤ p).
"""
function _fit(glr::GLR{L2Loss,<:L2R}, solver::Analytical, X, y)
	# full solve
	if !solver.iterative
		# augment X if appropriate
		X_ = augment_X(X, glr.fit_intercept)
		λ  = getscale(glr.penalty)
		# standard LS solution
		iszero(λ) && return X_ \ y
		# Ridge case -- form the Hat Matrix then solve
		H = Hermitian(X_'X_) + λ * I
		return cholesky!(H) \ X_'y
	end
	# Iterative case, note that there is no augmentation here
	# it is done implicitly in the application of the Hessian to
	# avoid copying X
	# The number of CG steps to convergence is at most `p`
	p = size(X, 2) + Int(glr.fit_intercept)
	max_cg_steps = min(solver.max_inner, p)
	# Form the Hessian map, cost of application H*v is O(np)
	Hm = LinearMap(Hv!(glr, X, y), p; ismutating=true, isposdef=true, issymmetric=true)
	b  = X'y
	glr.fit_intercept && (b = vcat(b, sum(y)))
	return cg(Hm, b; maxiter=max_cg_steps)
end
