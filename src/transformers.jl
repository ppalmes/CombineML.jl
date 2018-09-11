# Transformer definitions and implementations.
module Transformers

export Transformer,
       Learner,
       OneHotEncoder,
       Imputer,
       Pipeline,
       Wrapper,
       Identity,
       Baseline,
       PrunedTree, 
       RandomForest,
       DecisionStumpAdaboost,
       StandardScaler,
       PCA,
       VoteEnsemble, 
       StackEnsemble,
       BestLearner,
       SKLLearner,
       CRTLearner,
       fit!,
       transform!

## Obtain system details
import CombineML.System: LIB_SKL_AVAILABLE, LIB_CRT_AVAILABLE

# Include abstract types as convenience
using CombineML.Types
import CombineML.Types.fit!
import CombineML.Types.transform!

# Include atomic CombineML transformers
include(joinpath("combineml", "baseline.jl"))
using .BaselineMethods
include(joinpath("combineml", "transformers.jl"))
using .CombineMLTransformers

## Include Julia transformers
include(joinpath("julia", "decisiontree.jl"))
using .DecisionTreeWrapper
include(joinpath("julia", "mlbase.jl"))
using .MLBaseWrapper
include(joinpath("julia", "dimensionalityreduction.jl"))
using .DimensionalityReductionWrapper

# Include Python transformers
if LIB_SKL_AVAILABLE
  include(joinpath("python", "scikit_learn.jl"))
  using .ScikitLearnWrapper
end

# Include R transformers
if LIB_CRT_AVAILABLE
  include(joinpath("r", "caret.jl"))
  using .CaretWrapper
end

## Include aggregate transformers last, dependent on atomic transformers
include(joinpath("combineml", "ensemble.jl"))
using .EnsembleMethods

end # module
