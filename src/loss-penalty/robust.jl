export RobustLoss,
       HuberRho, Huber, AndrewsRho, Andrews,
       BisquareRho, Bisquare, LogisticRho, Logistic,
       FairRho, Fair, TalwarRho, Talwar, QuantileRho, Quantile

#=
In the non-penalised case:

   β⋆ = arg min ∑ ρ(yᵢ - ⟨xᵢ, β⟩)

where ρ is a weighing function such as, for instance, the pinball loss for
the quantile regression.

It is useful to define the following quantities:

   ψ(r) = ρ'(r)     (first derivative)
   ϕ(r) = ψ'(r)     (second derivative)
   ω(r) = ψ(r)/r    (weighing function used for IWLS), a threshold can be passed
                    to clip weights

Some refs:
- https://josephsalmon.eu/enseignement/UW/STAT593/QuantileRegression.pdf
=#

abstract type RobustRho end

# robust rho with only one parameter
abstract type RobustRho1P{δ} <: RobustRho end

struct RobustLoss{ρ} <: AtomicLoss where ρ <: RobustRho
   rho::ρ
end

(rl::RobustLoss)(Xβ::AVR, y::AVR) = rl(y .- Xβ)
(rl::RobustLoss)(r::AVR)          = rl.rho(r)


"""
$TYPEDEF

Huber weighing of the residuals corresponding to

``ρ(z) = z²/2``  if `|z|≤δ` and `ρ(z)=δ(|z|-δ/2)` otherwise.

Note: symmetric weighing.
"""
struct HuberRho{δ} <: RobustRho1P{δ}
   HuberRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end
Huber(δ::Real=1.0; delta::Real=δ) = HuberRho(delta)

(::HuberRho{δ})(r::AVR) where δ = begin
   ar = abs.(r)
   w  = ar .<= δ
   return sum( @. ifelse(w, r^2/2, δ * (ar - δ/2) ) )
end

ψ(::Type{HuberRho{δ}}   ) where δ = (r, w) -> r * w + δ * sign(r) * (1.0 - w)
ω(::Type{HuberRho{δ}}, _) where δ = (r, w) -> w + (δ / abs(r)) * (1.0 - w)
ϕ(::Type{HuberRho{δ}}   ) where δ = (r, w) -> w


"""
$TYPEDEF

Andrews weighing of the residuals corresponding to

``ρ(z) = -cos(πz/δ)/(π/δ)²`` if `|z|≤δ` and `ρ(δ)` otherwise.

Note: symmetric weighing.
"""
struct AndrewsRho{δ} <: RobustRho1P{δ}
   AndrewsRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end
Andrews(δ::Real=1.0; delta::Real=δ) = AndrewsRho(delta)

(::AndrewsRho{δ})(r::AVR) where δ = begin
   ar = abs.(r)
   w  = ar .<= δ
   c  = π/δ
   κ  = (δ/π)^2
   return sum( @. ifelse(w, -cos(c * r) * κ, κ) )
end

# Note, sinc(x) = sin(πx)/πx, well defined everywhere

ψ(::Type{AndrewsRho{δ}}   ) where δ = (r, w) -> (c = π/δ; w * sin(c * r) / c)
ω(::Type{AndrewsRho{δ}}, _) where δ = (r, w) -> w * sinc(r / δ)
ϕ(::Type{AndrewsRho{δ}}   ) where δ = (r, w) -> (cr = (π/δ) * r; w * cos(cr))


"""
$TYPEDEF

Bisquare weighing of the residuals corresponding to

``ρ(z) = δ²/6 (1-(1-(z/δ)²)³)`` if `|z|≤δ` and `δ²/6` otherwise.

Note: symmetric weighing.
"""
struct BisquareRho{δ} <: RobustRho1P{δ}
   BisquareRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end
Bisquare(δ::Real=1.0; delta::Real=δ) = BisquareRho(delta)

(::BisquareRho{δ})(r::AVR) where δ = begin
   ar = abs.(r)
   w  = ar .<= δ
   κ  = δ^2/6
   return sum( @. ifelse(w, κ * (1 - (1 - (r / δ)^2)^3), κ) )
end

ψ(::Type{BisquareRho{δ}}   ) where δ = (r, w) -> w * r * (1.0 - (r / δ)^2)^2
ω(::Type{BisquareRho{δ}}, _) where δ = (r, w) -> w * (1.0 - (r / δ)^2)^2
ϕ(::Type{BisquareRho{δ}}   ) where δ = (r, w) -> (sr = r / δ; w * (1.0 + 5sr^4 - 6sr^2))

"""
$TYPEDEF

Logistic weighing of the residuals corresponding to

``ρ(z) = δ² log(cosh(z/δ))``

Note: symmetric weighing.
"""
struct LogisticRho{δ} <: RobustRho1P{δ}
   LogisticRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end
Logistic(δ::Real=1.0; delta::Real=δ) = LogisticRho(delta)

(::LogisticRho{δ})(r::AVR) where δ = begin
   return sum( @. δ^2 * log(cosh(r / δ)) )
end

# similar to sinc, to avoid NaNs if tanh(0)/0 (lim is 1.0)
tanhc(x::Real) = ifelse(iszero(x), one(x), tanh(x)/x)

ψ(::Type{LogisticRho{δ}}   ) where δ = (r, _) -> δ * tanh(r / δ)
ω(::Type{LogisticRho{δ}}, _) where δ = (r, _) -> tanhc(r / δ)
ϕ(::Type{LogisticRho{δ}}   ) where δ = (r, w) -> sech(r / δ)^2


"""
$TYPEDEF

Fair weighing of the residuals corresponding to

``ρ(z) = δ² (|z|/δ - log(1+|z|/δ))``

Note: symmetric weighing.
"""
struct FairRho{δ} <: RobustRho1P{δ}
   FairRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end
Fair(δ::Real=1.0; delta::Real=δ) = FairRho(delta)

(::FairRho{δ})(r::AVR) where δ = begin
   sr = @. abs(r) / δ
   return sum( @. δ^2 * (sr - log1p(sr)) )
end

ψ(::Type{FairRho{δ}}   ) where δ = (r, _) -> δ * r / (abs(r) + δ)
ω(::Type{FairRho{δ}}, _) where δ = (r, _) -> δ / (abs.(r) + δ)
ϕ(::Type{FairRho{δ}}   ) where δ = (r, _) -> δ^2 / (abs.(r) + δ)^2


"""
$TYPEDEF

Talwar weighing of the residuals corresponding to

``ρ(z) = z²/2`` if `|z|≤δ` and `ρ(z)=ρ(δ)` otherwise.

Note: symmetric weighing.
"""
struct TalwarRho{δ} <: RobustRho1P{δ}
   TalwarRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end
Talwar(δ::Real=1.0; delta::Real=δ) = TalwarRho(delta)

(::TalwarRho{δ})(r::AVR) where δ = begin
   w = @. abs(r) <= δ
   return sum( @. ifelse(w, r^2 / 2, δ^2/2) )
end

ψ(::Type{TalwarRho{δ}}   ) where δ = (r, w) -> w * r
ω(::Type{TalwarRho{δ}}, _) where δ = (_, w) -> w
ϕ(::Type{TalwarRho{δ}}   ) where δ = (_, w) -> w


"""
$TYPEDEF

Quantile regression weighing of the residuals corresponding to

``ρ(z) = -z(δ - 1(z>=0))``

Note: asymetric weighing, the "-" sign is because similar libraries like
quantreg for instance define the residual as `y-Xθ` while we do the opposite
(out of convenience for gradients etc).
"""
struct QuantileRho{δ} <: RobustRho1P{δ}
   QuantileRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end

Quantile(δ::Real=1.0; delta::Real=δ) = QuantileRho(delta)

(::QuantileRho{δ})(r::AVR) where δ = begin
   return sum( @. -r * (δ - (r >= 0)) )
end

ψ(::Type{QuantileRho{δ}}   ) where δ = (r, _) -> ((r >= 0.0) - δ)
ω(::Type{QuantileRho{δ}}, τ) where δ = (r, _) -> ((r >= 0.0) - δ) / clip(-r, τ)
ϕ(::Type{QuantileRho{δ}}   ) where δ = (_, _) -> error("Newton(CG) not available for Quantile Reg.")
