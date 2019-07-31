# Dimensionality Reduction transformers.
module DimensionalityReductionWrapper

using DataFrames
using CombineML.Types
import CombineML.Types.fit!
import CombineML.Types.transform!
using CombineML.Util
import MultivariateStats
import MultivariateStats: fit, transform

export PCA,
       fit!,
       transform!

## Principal Component Analysis rotation
## on features.
## Features ordered by maximal variance descending.
##
## Fails if zero-variance feature exists.
mutable struct PCA <: Transformer
  model
  options

  function PCA(options=Dict())
    default_options = Dict(
      #:center => true,
      #:scale => true
      :pratio => 1.0
    )
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(p::PCA, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances=convert(Matrix,features)
  _pratio=1.0
  _maxoutdim=size(xinstances')[1]
  if haskey(p.options,:pratio)
    _pratio=p.options[:pratio]
  end
  if haskey(p.options,:maxoutdim)
    _maxoutdim=convert(Int64,p.options[:maxoutdim])
  end
  pca_model =fit(MultivariateStats.PCA,xinstances',pratio=_pratio,maxoutdim=_maxoutdim)
  p.model = pca_model
end

function transform!(p::PCA, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances = convert(Matrix,features)
  res=transform(p.model,xinstances')
  return res'
end

end # module
