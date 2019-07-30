# Decision trees as found in DecisionTree Julia package.
module DecisionTreeWrapper

using DataFrames
using CombineML.Types
import CombineML.Types.fit!
import CombineML.Types.transform!
using CombineML.Util

import DecisionTree
DT = DecisionTree

export PrunedTree, 
       RandomForest,
       DecisionStumpAdaboost,
       fit!, 
       transform!

# Pruned CART decision tree.
mutable struct PrunedTree <: Learner
  model
  options
  
  function PrunedTree(options=Dict())
    default_options = Dict(
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_options => Dict(
        # Merge leaves having >= purity_threshold CombineMLd purity.
        :purity_threshold => 1.0,
        # Maximum depth of the decision tree (default: no maximum).
        :max_depth => -1,
        # Minimum number of samples each leaf needs to have.
        :min_samples_leaf => 1,
        # Minimum number of samples in needed for a split.
        :min_samples_split => 2,
        # Minimum purity needed for a split.
        :min_purity_increase => 0.0
      )
    )
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(tree::PrunedTree, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances=convert(Matrix,features)
  impl_options = tree.options[:impl_options]
  tree.model = DT.build_tree(
    labels,
    xinstances,
    0, # num_subfeatures (keep all)
    impl_options[:max_depth],
    impl_options[:min_samples_leaf],
    impl_options[:min_samples_split],
    impl_options[:min_purity_increase])
  tree.model = DT.prune_tree(tree.model, impl_options[:purity_threshold])
end

function transform!(tree::PrunedTree, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances=convert(Matrix,features)
  return DT.apply_tree(tree.model, xinstances)
end

# Random forest (CART).
mutable struct RandomForest <: Learner
  model
  options
  
  function RandomForest(options=Dict())
    default_options = Dict(
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_options => Dict(
        # Number of features to train on with trees (default: 0, keep all).
        :num_subfeatures => 0,
        # Number of trees in forest.
        :num_trees => 10,
        # Proportion of trainingset to be used for trees.
        :partial_sampling => 0.7,
        # Maximum depth of each decision tree (default: no maximum).
        :max_depth => -1
      )
    )
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(forest::RandomForest, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances=convert(Matrix,features)
  # Set training-dependent options
  impl_options = forest.options[:impl_options]
  # Build model
  forest.model = DT.build_forest(
    labels, 
    xinstances,
    impl_options[:num_subfeatures],
    impl_options[:num_trees],
    impl_options[:partial_sampling],
    impl_options[:max_depth]
  )
end

function transform!(forest::RandomForest, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances = convert(Matrix,features)
  return DT.apply_forest(forest.model, xinstances)
end

# Adaboosted decision stumps.
mutable struct DecisionStumpAdaboost <: Learner
  model
  options
  
  function DecisionStumpAdaboost(options=Dict())
    default_options = Dict(
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_options => Dict(
        # Number of boosting iterations.
        :num_iterations => 7
      )
    )
    new(nothing, nested_dict_merge(default_options, options))
  end
end

function fit!(adaboost::DecisionStumpAdaboost, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances = convert(Matrix,features)

  # NOTE(svs14): Variable 'model' renamed to 'ensemble'.
  #              This differs to DecisionTree
  #              official documentation to avoid confusion in variable
  #              naming within CombineML.
  ensemble, coefficients = DT.build_adaboost_stumps(
    labels, xinstances, adaboost.options[:impl_options][:num_iterations]
  )
  adaboost.model = Dict(
    :ensemble => ensemble,
    :coefficients => coefficients
  )
end

function transform!(adaboost::DecisionStumpAdaboost, features::T) where {T<:Union{Vector,Matrix,DataFrame}}
  xinstances = convert(Matrix,features)
  return DT.apply_adaboost_stumps(
    adaboost.model[:ensemble], adaboost.model[:coefficients], xinstances
  )
end

end # module
