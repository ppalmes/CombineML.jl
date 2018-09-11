# Baseline methods.
module BaselineMethods

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

function fit!(bl::Baseline, instances::Matrix, labels::Vector)
  bl.model = bl.options[:strategy](labels)
end

function transform!(bl::Baseline, instances::Matrix)
  return fill(bl.model, size(instances, 1))
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

function fit!(id::Identity, instances::Matrix, labels::Vector)
  nothing
end

function transform!(id::Identity, instances::Matrix)
  return instances
end

end # module
