# CombineML types.
module Types
using DataFrames

export Transformer,
       Learner,
       TestLearner,
       fit!,
       transform!

# All transformer types must have implementations 
# of function `fit!` and `transform!`.
abstract type Transformer end

# Learner abstract type which all machine learners implement.
abstract type Learner <: Transformer end

# Test learner. 
# Used to separate production learners from test.
abstract type TestLearner <: Learner end

# Trains transformer on provided instances and labels.
#
# @param transformer Target transformer.
# @param instances Training instances.
# @param labels Training labels.
function fit!(transformer::Transformer, xinstances::T, labels::Vector) where {T <: Union{Vector,Matrix,DataFrame}}
  error(typeof(transformer), " does not implement fit!")
end

# Trains transformer on provided instances and labels.
#
# @param transformer Target transformer.
# @param instances Original instances.
# @return Transformed instances.
function transform!(transformer::Transformer, xinstances::T) where {T <: Union{Vector,Matrix,DataFrame}}
  error(typeof(transformer), " does not implement transform!")
end

end # module
