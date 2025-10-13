# RandESC
## Solve eigenvalue problems using randomized algorithms

Install the package (creating a Julia environment in a different folder with RandESC in it):
```
# enter Julia REPL console:
julia
julia> import Pkg
julia> Pkg.develop(path="/path/to/RandESC/)
```

Run tests:
```
julia --project=. -e 'using Pkg; Pkg.test()'
```
This has to be done from inside the source directory of this repository, otherwise the the path must be specified in the option `--project=`
