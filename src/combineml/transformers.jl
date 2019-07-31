# Transformers provided by CombineML.
module CombineMLTransformers

using DataFrames
using Statistics
using CombineML.Types
import CombineML.Types.fit!
import CombineML.Types.transform!
using CombineML.Util

export OneHotEncoder,
       Imputer,
       Pipeline,
       Wrapper,
       fit!,
       transform!

# Transforms instances with nominal features into one-hot form 
# and coerces the instance matrix to be of element type Float64.
mutable struct OneHotEncoder <: Transformer
  model
  options

  function OneHotEncoder(options=Dict())
    default_options = Dict(
      # Nominal columns
      :nominal_columns => nothing,
      # Nominal column values map. Key is column index, value is list of
      # possible values for that column.
      :nominal_column_values_map => nothing
    )
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(ohe::OneHotEncoder, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances=convert(Matrix,features)
  # Obtain nominal columns
  nominal_columns = ohe.options[:nominal_columns]
  if nominal_columns == nothing
    nominal_columns = find_nominal_columns(xinstances)
  end

  # Obtain unique values for each nominal column
  nominal_column_values_map = ohe.options[:nominal_column_values_map]
  if nominal_column_values_map == nothing
    nominal_column_values_map = Dict{Int, Any}()
    for column in nominal_columns
      nominal_column_values_map[column] = unique(xinstances[:, column])
    end
  end

  # Create model
  ohe.model = Dict(
    :nominal_columns => nominal_columns,
    :nominal_column_values_map => nominal_column_values_map
  )
end

function transform!(ohe::OneHotEncoder, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances=convert(Matrix,features)
  nominal_columns = ohe.model[:nominal_columns]
  nominal_column_values_map = ohe.model[:nominal_column_values_map]

  # Create new transformed instance matrix of type Float64
  num_rows = size(xinstances, 1)
  num_columns = (size(xinstances, 2) - length(nominal_columns)) 
  if !isempty(nominal_column_values_map)
    num_columns += sum(map(x -> length(x), values(nominal_column_values_map)))
  end
  transformed_instances = zeros(Float64, num_rows, num_columns)

  # Fill transformed instance matrix
  col_start_index = 1
  for column in 1:size(xinstances, 2)
    if !in(column, nominal_columns)
      transformed_instances[:, col_start_index] = xinstances[:, column]
      col_start_index += 1
    else
      col_values = nominal_column_values_map[column]
      for row in 1:size(xinstances, 1)
        entry_value = xinstances[row, column]
        entry_value_index = findfirst(isequal(entry_value),col_values)
        if entry_value_index == 0
          warn("Unseen value found in OneHotEncoder,
                for entry ($row, $column) = $(entry_value). 
                Patching value to $(col_values[1]).")
          entry_value_index = 1
        end
        entry_column = (col_start_index - 1) + entry_value_index
        transformed_instances[row, entry_column] = 1
      end
      col_start_index += length(nominal_column_values_map[column])
    end
  end

  return transformed_instances
end

# Finds all nominal columns.
# 
# Nominal columns are those that do not have Real type nor
# do all their elements correspond to Real.
function find_nominal_columns(features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances=convert(Matrix,features)
  nominal_columns = Int[]
  for column in 1:size(xinstances, 2)
    col_eltype = infer_eltype(xinstances[:, column])
    if !<:(col_eltype, Real)
      push!(nominal_columns, column)
    end
  end
  return nominal_columns
end


# Imputes NaN values from Float64 features.
mutable struct Imputer <: Transformer
  model
  options

  function Imputer(options=Dict())
    default_options = Dict(
      # Imputation strategy.
      # Statistic that takes a vector such as mean or median.
      :strategy => mean
    )
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(imp::Imputer, xinstances::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  imp.model = imp.options
end

function transform!(imp::Imputer, features::T)  where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances=convert(Matrix,features)
  new_instances = copy(xinstances)
  strategy = imp.model[:strategy]

  for column in 1:size(xinstances, 2)
    column_values = xinstances[:, column]
    col_eltype = infer_eltype(column_values)

    if <:(col_eltype, Real)
      na_rows = map(x -> isnan(x), column_values)
      if any(na_rows)
        fill_value = strategy(column_values[.!na_rows])
        new_instances[na_rows, column] .= fill_value
      end
    end
  end

  return new_instances
end


# Chains multiple transformers in sequence.
mutable struct Pipeline <: Transformer
  model
  options

  function Pipeline(options=Dict())
    default_options = Dict(
      # Transformers as list to chain in sequence.
      :transformers => [OneHotEncoder(), Imputer()],
      # Transformer options as list applied to same index transformer.
      :transformer_options => nothing
    )
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(pipe::Pipeline, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances=convert(Matrix,features)
  transformers = pipe.options[:transformers]
  transformer_options = pipe.options[:transformer_options]

  current_instances = xinstances
  new_transformers = Transformer[]
  for t_index in 1:length(transformers)
    transformer = create_transformer(transformers[t_index], transformer_options)
    push!(new_transformers, transformer)
    fit!(transformer, current_instances, labels)
    current_instances = transform!(transformer, current_instances)
  end

  pipe.model = Dict(
      :transformers => new_transformers,
      :transformer_options => transformer_options
  )
end

function transform!(pipe::Pipeline, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances = convert(Matrix,features)
  transformers = pipe.model[:transformers]

  current_instances = xinstances
  for t_index in 1:length(transformers)
    transformer = transformers[t_index]
    current_instances = transform!(transformer, current_instances)
  end

  return current_instances
end


# Wraps around an CombineML transformer.
mutable struct Wrapper <: Transformer
  model
  options

  function Wrapper(options=Dict())
    default_options = Dict(
      # Transformer to call.
      :transformer => OneHotEncoder(),
      # Transformer options.
      :transformer_options => nothing
    )
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(wrapper::Wrapper, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances=convert(Matrix,features)
  transformer_options = wrapper.options[:transformer_options]
  transformer = create_transformer(
    wrapper.options[:transformer],
    transformer_options
  )

  if transformer_options != nothing
    transformer_options = 
      nested_dict_merge(transformer.options, transformer_options)
  end
  fit!(transformer, xinstances, labels)

  wrapper.model = Dict(
    :transformer => transformer,
    :transformer_options => transformer_options
  )
end

function transform!(wrapper::Wrapper, xinstances::T) where {T<:Union{Vector,Matrix,DataFrame}}
  transformer = wrapper.model[:transformer]
  return transform!(transformer, xinstances)
end

end # module
