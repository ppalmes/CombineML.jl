# MLBase transformers.
module MLBaseWrapper

importall CombineML.Types
importall CombineML.Util

import MLBase: Standardize, estimate, transform

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

function fit!(st::StandardScaler, instances::Matrix, labels::Vector)
  st_transform = estimate(Standardize, instances'; st.options...)
  st.model = Dict(
    :standardize_transform => st_transform
  )
end

function transform!(st::StandardScaler, instances::Matrix)
  st_transform = st.model[:standardize_transform]
  transposed_instances = instances'
  return transform(st_transform, transposed_instances)'
end

end # module
