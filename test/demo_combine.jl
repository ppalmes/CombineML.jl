#using CombineML
#Base.compilecache("CombineML")

# http://topepo.github.io/caret/available-models.html
# https://github.com/svs14/Orchestra.jl

import CombineML

try
  import RDatasets
catch
  using Pkg
  Pkg.add("RDatasets")
  import RDatasets
end

using Distributed

addprocs()

@everywhere using Random
@everywhere using Statistics
@everywhere import CombineML
@everywhere import CombineML.Util
@everywhere import CombineML.Transformers
@everywhere import RDatasets

@everywhere CU=CombineML.Util
@everywhere CT=CombineML.Transformers
@everywhere RD=RDatasets

subtypes(CT.Transformer)

subtypes(CT.Learner)

@everywhere function predict(learner)
    dataset = RD.dataset("datasets", "iris")
    features = convert(Matrix,dataset[:, 1:(end-1)])
    labels = convert(Array,dataset[:, end])
    # Split into training and test sets
    (train_ind, test_ind) = CU.holdout(size(features, 1), 0.3)
    # Create pipeline
    pipeline = CT.Pipeline(
      Dict(
        :transformers => [
          CT.OneHotEncoder(), # Encodes nominal features into numeric
          CT.Imputer(), # Imputes NA values
          CT.StandardScaler(), # Standardizes features 
          learner # Predicts labels on instances
        ]
      )
    )
    # Train
    CT.fit!(pipeline, features[train_ind, :], labels[train_ind]);
    # Predict
    predictions = CT.transform!(pipeline, features[test_ind, :]);
    # Assess predictions
    result = CU.score(:accuracy, labels[test_ind], predictions)
    return result
end
# Learner with default settings
learner = CT.PrunedTree()
predict(learner)

# Learner with some of the default settings overridden
learner = CT.PrunedTree(
  Dict(:impl_options => Dict(:purity_threshold => 1.0))
)
predict(learner)

learner = CT.PrunedTree(
  Dict(
    :output => :class,
    :impl_options => Dict(:purity_threshold => 1.0) 
  )
)
predict(learner)

learner = CT.VoteEnsemble(
  Dict(
    :output => :class,
    :learners => [CT.PrunedTree(), CT.DecisionStumpAdaboost(), CT.RandomForest()]
  )
)
predict(learner)

learner = CT.BestLearner(
  Dict(
    :output => :class,
    #:partition_generator => (instances, labels) -> CU.kfold(size(instances, 1), 5),
    #:selection_function => (learner_partition_scores) -> findmax(mean(learner_partition_scores, 2))[2],      
    :score_type => Real,
    :learners => [CT.PrunedTree(), CT.DecisionStumpAdaboost(), CT.RandomForest()],
    :learner_options_grid => nothing
  )
)
predict(learner)

@everywhere rf = CT.RandomForest(
  Dict(
    :output => :class,
    :impl_options => Dict(
      :num_trees => 100,
      :partial_sampling => 0.7
    )
  )
)
predict(rf)

remotecall_fetch(predict,3,rf)

@everywhere ada = CT.DecisionStumpAdaboost(
  Dict(
    :output => :class,
    :impl_options => Dict(:num_iterations => 7)
  )
)
predict(ada)


if CombineML.System.LIB_SKL_AVAILABLE
  @everywhere sk_learner = CT.SKLLearner(
   Dict(
        :output => :class,
        #:learner => "BaggingClassifier",
        #:learner => "ExtraTreesClassifier",
        :learner => "GradientBoostingClassifier",
        :impl_options => Dict()
       )
  )
else
  @everywhere sk_learner = CT.Baseline()
end
predict(sk_learner)

if CombineML.System.LIB_CRT_AVAILABLE
  @everywhere crt_learner = CT.CRTLearner(
    Dict(
      :output => :class,
      #:learner => "rf",
      :learner => "svmLinear2",
      #:learner => "rpart",
      :impl_options => Dict()
    )
  )
else
  @everywhere crt_learner = CT.Baseline()
end
predict(crt_learner)

@everywhere stacklearner = CT.StackEnsemble(Dict(
    :output => :class,
    :learners => [CT.PrunedTree(), CT.RandomForest(),CT.DecisionStumpAdaboost(),rf,crt_learner,sk_learner], 
    :stacker => 
       CT.RandomForest(
          Dict(
             :output => :class,
             :impl_options => Dict(
               :num_trees => 20,
               :partial_sampling => 0.7
             )
          )
       ),
    #:stacker_training_proportion => 0.3,
    :keep_original_features => true
))
predict(stacklearner)


@everywhere bestlearner = CT.BestLearner(
  Dict(
    :learners => [stacklearner,CT.PrunedTree(), CT.RandomForest(),CT.DecisionStumpAdaboost(),rf,sk_learner,crt_learner], 
    :output => :class,
    :score_type => Real,
    :learner_options_grid => nothing
  )
)
predict(bestlearner)

@everywhere votelearner = CT.VoteEnsemble(
  Dict(
    :learners => [stacklearner,CT.PrunedTree(), CT.RandomForest(),CT.DecisionStumpAdaboost(),rf,crt_learner,sk_learner], 
    :output => :class,
  )
)
predict(votelearner)

# All learners are called in the same way.
@everywhere stackstacklearner = CT.StackEnsemble(
  Dict(
    :learners => [stacklearner,votelearner,CT.PrunedTree(), CT.RandomForest(),CT.DecisionStumpAdaboost(),rf,crt_learner,sk_learner], 
    #:keep_original_features => false,
    #:stacker_training_proportion => 0.3,
    :stacker => 
       CT.RandomForest(
          Dict(
             :output => :class,
             :impl_options => Dict(
               :num_trees => 20,
               :partial_sampling => 0.7
             )
          )
       )
  )
)
predict(stackstacklearner)


acc=@distributed (vcat) for i=1:5
    res=predict(stacklearner)
    println(round(res))
    res
end
println("Acc=",round(mean(acc))," +/- ",round(std(acc)))

using DataFrames
function main(trials)
    learners=Dict(:votelearner=>votelearner,:stacklearner=>stacklearner,
                  :stackstacklearner=>stackstacklearner,
                  :caretlearner=>crt_learner,
                  :scikitlearner=>sk_learner,
                  :juliaprunetree=>CT.PrunedTree(), 
                  :juliaadaboost=>CT.DecisionStumpAdaboost(), 
                  :baseline=>CT.Baseline(),
                  :juliarandomforest=>CT.RandomForest())
    models=collect(keys(learners))
    ctable=@distributed (vcat) for model in models
        acc=@distributed (vcat) for i=1:trials
            res=predict(learners[model])
            println(model," => ",round(res))
            res
        end
        [model round(mean(acc)) round(std(acc)) length(acc)]
    end
    sorted=sort(DataFrame(ctable),:x2,rev=true)
    rename!(sorted,Dict(:x1=>:model,:x2=>:mean_acc,:x3=>:std_acc,:x4=>:trials))
    return sorted
end
res=main(10) 
