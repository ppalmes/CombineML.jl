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
importall CombineML.Types

# Include atomic CombineML transformers
include(joinpath("combineml", "baseline.jl"))
importall .BaselineMethods
include(joinpath("combineml", "transformers.jl"))
importall .CombineMLTransformers

## Include Julia transformers
include(joinpath("julia", "decisiontree.jl"))
importall .DecisionTreeWrapper
include(joinpath("julia", "mlbase.jl"))
importall .MLBaseWrapper
include(joinpath("julia", "dimensionalityreduction.jl"))
importall .DimensionalityReductionWrapper

# Include Python transformers
if LIB_SKL_AVAILABLE
  include(joinpath("python", "scikit_learn.jl"))
  importall .ScikitLearnWrapper
end

# Include R transformers
if LIB_CRT_AVAILABLE
  include(joinpath("r", "caret.jl"))
  importall .CaretWrapper
end

## Include aggregate transformers last, dependent on atomic transformers
include(joinpath("combineml", "ensemble.jl"))
importall .EnsembleMethods

end # module
