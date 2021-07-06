using Documenter, MLJLinearModels

makedocs(
    modules = [MLJLinearModels],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        ),
    sitename = "MLJLinearModels.jl",
    authors = "Thibaut Lienart, and contributors.",
    pages = [
        "Home" => "index.md",
        "Quick start"  => "quickstart.md",
        "Models" => "models.md",
        "Solvers" => "solvers.md",
        "API" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/JuliaAI/MLJLinearModels.jl"
)
