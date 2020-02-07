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
    ]
)

deploydocs(
    repo = "github.com/alan-turing-institute/MLJLinearModels.jl"
)
