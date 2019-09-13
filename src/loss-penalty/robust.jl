export RobustLoss,
       HuberRho, Huber, AndrewsRho, Andrews,
       BisquareRho, Bisquare, LogisticRho, Logistic,
       FairRho, Fair, TalwarRho, Talwar, QuantileRho, Quantile

abstract type RobustRho end

abstract type RobustRho1P{δ} <: RobustRho end # one parameter

struct RobustLoss{ρ} <: AtomicLoss where ρ <: RobustRho
   rho::ρ
end

(rl::RobustLoss)(x::AVR, y::AVR) = rl(x .- y)
(rl::RobustLoss)(r::AVR) = rl.rho(r)

# ψ(r) = ρ'(r)     (first derivative)
# ω(r) = ψ(r)/r    (weighing function) a threshold can be passed to clip weights
# ϕ(r) = ψ'(r)     (second derivative)

"""
$TYPEDEF

Huber weighing of the residuals corresponding to

``ρ(z) = z²/2``  if `|z|≤δ` and `ρ(z)=δ(|z|-δ/2)` otherwise.
"""
struct HuberRho{δ} <: RobustRho1P{δ}
   HuberRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end
Huber(δ::Real=1.0; delta::Real=δ) = HuberRho(delta)

(::HuberRho{δ})(r::AVR) where δ = begin
   ar = abs.(r)
   w  = ar .<= δ
   return sum( r.^2/2 .* w .+ δ .* (ar .- δ/2) .* .!w )
end

ψ(::Type{HuberRho{δ}}   ) where δ = (r, w) -> r * w + δ * sign(r) * (1.0 - w)
ω(::Type{HuberRho{δ}}, _) where δ = (r, w) -> w + (δ / abs(r)) * (1.0 - w)
ϕ(::Type{HuberRho{δ}}   ) where δ = (r, w) -> w


"""
$TYPEDEF

Andrews weighing of the residuals corresponding to

``ρ(z) = -cos(πz/δ)/(π/δ)²`` if `|z|≤δ` and `ρ(δ)` otherwise.
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
   return sum( -cos.(c .* r) .* κ .* w .+ κ .* .!w )
end

# Note, sinc(x) = sin(πx)/πx, well defined everywhere

ψ(::Type{AndrewsRho{δ}}   ) where δ = (r, w) -> (c = π/δ; w * sin(c * r) / c)
ω(::Type{AndrewsRho{δ}}, _) where δ = (r, w) -> w * sinc(r / δ)
ϕ(::Type{AndrewsRho{δ}}   ) where δ = (r, w) -> (cr = (π/δ) * r; w * cos(cr))


"""
$TYPEDEF

Bisquare weighing of the residuals corresponding to

``ρ(z) = δ²/6 (1-(1-(z/δ)²)³)`` if `|z|≤δ` and `δ²/6` otherwise.
"""
struct BisquareRho{δ} <: RobustRho1P{δ}
   BisquareRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end
Bisquare(δ::Real=1.0; delta::Real=δ) = BisquareRho(delta)

(::BisquareRho{δ})(r::AVR) where δ = begin
   ar = abs.(r)
   w  = ar .<= δ
   κ  = δ^2/6
   return sum( κ * (1.0 .- (1.0 .- (r ./ δ).^2).^3) .* w + κ .* .!w )
end

ψ(::Type{BisquareRho{δ}}   ) where δ = (r, w) -> w * r * (1.0 - (r / δ)^2)^2
ω(::Type{BisquareRho{δ}}, _) where δ = (r, w) -> w * (1.0 - (r / δ)^2)^2
ϕ(::Type{BisquareRho{δ}}   ) where δ = (r, w) -> (sr = r / δ; w * (1.0 + 5sr^4 - 6sr^2))

"""
$TYPEDEF

Logistic weighing of the residuals corresponding to

``ρ(z) = δ² log(cosh(z/δ))``
"""
struct LogisticRho{δ} <: RobustRho1P{δ}
   LogisticRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end
Logistic(δ::Real=1.0; delta::Real=δ) = LogisticRho(delta)

(::LogisticRho{δ})(r::AVR) where δ = begin
   return sum( δ^2 .* log.(cosh.(r ./ δ)) )
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
"""
struct FairRho{δ} <: RobustRho1P{δ}
   FairRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end
Fair(δ::Real=1.0; delta::Real=δ) = FairRho(delta)

(::FairRho{δ})(r::AVR) where δ = begin
   sr = abs.(r) ./ δ
   return sum( δ^2 .* (sr .- log1p.(sr)) )
end

ψ(::Type{FairRho{δ}}   ) where δ = (r, _) -> δ * r / (abs(r) + δ)
ω(::Type{FairRho{δ}}, _) where δ = (r, _) -> δ / (abs.(r) + δ)
ϕ(::Type{FairRho{δ}}   ) where δ = (r, _) -> δ^2 / (abs.(r) + δ)^2


"""
$TYPEDEF

Talwar weighing of the residuals corresponding to

``ρ(z) = z²/2`` if `|z|≤δ` and `ρ(z)=ρ(δ)` otherwise.
"""
struct TalwarRho{δ} <: RobustRho1P{δ}
   TalwarRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end
Talwar(δ::Real=1.0; delta::Real=δ) = TalwarRho(delta)

(::TalwarRho{δ})(r::AVR) where δ = begin
   w = abs.(r) .<= δ
   return sum( r.^2 ./ 2 .* w .+ δ^2/2 .* .!w)
end

ψ(::Type{TalwarRho{δ}}   ) where δ = (r, w) -> w * r
ω(::Type{TalwarRho{δ}}, _) where δ = (_, w) -> w
ϕ(::Type{TalwarRho{δ}}   ) where δ = (_, w) -> w


"""
$TYPEDEF

Quantile regression weighing of the residuals corresponding to

``ρ(z) = z(δ - 1(z<0))``
"""
struct QuantileRho{δ} <: RobustRho1P{δ}
   QuantileRho(δ::Real=1.0; delta::Real=δ) = new{delta}()
end

Quantile(δ::Real=1.0; delta::Real=δ) = QuantileRho(delta)

(::QuantileRho{δ})(r::AVR) where δ = begin
   return sum( r .* (δ .- (r .<= 0.0)) )
end

ψ(::Type{QuantileRho{δ}}   ) where δ = (r, _) -> (δ - (r <= 0.0))
ω(::Type{QuantileRho{δ}}, τ) where δ = (r, _) -> (δ - (r <= 0.0)) / clip(r, τ)
ϕ(::Type{QuantileRho{δ}}   ) where δ = (_, _) -> error("Newton(CG) not available for Quantile Reg.")
