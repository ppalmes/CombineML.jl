# Baseline methods.
module BaselineMethods

using DataFrames
using CombineML.Types
import CombineML.Types.fit!
import CombineML.Types.transform!
using CombineML.Util
import StatsBase: mode

export Baseline,
       Identity,
       fit!,
       transform!

# Baseline learner that by default assigns the most frequent label.
mutable struct Baseline <: Learner
  model
  options

  function Baseline(options=Dict())
    default_options = Dict( 
      # Output to train against
      # (:class).
      :output => :class,
      # Label assignment strategy.
      # Function that takes a label vector and returns the required output.
      :strategy => mode
    )
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(bl::Baseline, xinstances::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  bl.model = bl.options[:strategy](labels)
end

function transform!(bl::Baseline, xinstances::T) where {T<:Union{Vector,Matrix,DataFrame}}
  return fill(bl.model, size(xinstances, 1))
end


# Identity transformer passes the instances as is.
mutable struct Identity <: Transformer
  model
  options

  function Identity(options=Dict())
    default_options = Dict{Symbol, Any}()
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(id::Identity, xinstances::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  nothing
end

function transform!(id::Identity, xinstances::T) where {T<:Union{Vector,Matrix,DataFrame}}
  return xinstances
end

end # module
