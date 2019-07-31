# MLBase transformers.
module MLBaseWrapper

using DataFrames
using CombineML.Types
import CombineML.Types.fit!
import CombineML.Types.transform!
using CombineML.Util

include("standardize.jl")

import .MStandardize: Standardize, estimate, transform

export StandardScaler,
       fit!,
       transform!

# Standardizes each feature using (X - mean) / stddev.
# Will produce NaN if standard deviation is zero.
mutable struct StandardScaler <: Transformer
  model
  options

  function StandardScaler(options=Dict())
    default_options = Dict( 
      :center => true,
      :scale => true
    )
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(st::StandardScaler, xinstances::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  features = convert(Matrix,xinstances)
  st_transform = estimate(Standardize, Array(features'); st.options...)
  st.model = Dict(
    :standardize_transform => st_transform
  )
end

function transform!(st::StandardScaler, xinstances::T)  where {T<:Union{Vector,Matrix,DataFrame}}
  features = convert(Matrix,xinstances)
  st_transform = st.model[:standardize_transform]
  transposed_instances = Array(features')
  return Array(transform(st_transform, transposed_instances)')
end

end # module
