#using Combine
#Base.compilecache("Combine")

# http://topepo.github.io/caret/available-models.html
# https://github.com/svs14/Orchestra.jl

addprocs()

@everywhere import Combine.Util
@everywhere import Combine.Transformers
@everywhere import RDatasets

@everywhere CU=Combine.Util
@everywhere CT=Combine.Transformers
@everywhere RD=RDatasets

subtypes(CT.Transformer)

subtypes(CT.Learner)

@everywhere function predict(learner)
    dataset = RD.dataset("datasets", "iris")
    instances = Array(dataset[:, 1:(end-1)])
    labels = Array(dataset[:, end])
    # Split into training and test sets
    (train_ind, test_ind) = CU.holdout(size(instances, 1), 0.3)
    # Create pipeline
    pipeline = CT.Pipeline(
      Dict(
        :transformers => [
          CT.OneHotEncoder(), # Encodes nominal features into numeric
          CT.Imputer(), # Imputes NA values
          #StandardScaler(), # Standardizes features 
          learner # Predicts labels on instances
        ]
      )
    )
    # Train
    CT.fit!(pipeline, instances[train_ind, :], labels[train_ind]);
    # Predict
    predictions = CT.transform!(pipeline, instances[test_ind, :]);
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
      :num_subfeatures => nothing,
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


@everywhere sk_learner = CT.SKLLearner(
  Dict(
    :output => :class,
    #:learner => "BaggingClassifier",
    #:learner => "ExtraTreesClassifier",
    :learner => "GradientBoostingClassifier",
    :impl_options => Dict()
  )
)
predict(sk_learner)

@everywhere crt_learner = CT.CRTLearner(
  Dict(
    :output => :class,
    #:learner => "rf",
    :learner => "svmLinear2",
    #:learner => "rpart",
    :impl_options => Dict()
  )
)
predict(crt_learner)

@everywhere stacklearner = CT.StackEnsemble(Dict(
    :output => :class,
    :learners => [sk_learner,crt_learner,CT.PrunedTree(), CT.RandomForest(),CT.DecisionStumpAdaboost(),rf], 
    :stacker => 
       CT.RandomForest(
          Dict(
             :output => :class,
             :impl_options => Dict(
               :num_subfeatures => nothing,
               :num_trees => 200,
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
    :learners => [stacklearner,sk_learner,crt_learner,CT.PrunedTree(), CT.RandomForest(),CT.DecisionStumpAdaboost(),rf], 
    :output => :class,
    :score_type => Real,
    :learner_options_grid => nothing
  )
)
predict(bestlearner)

@everywhere votelearner = CT.VoteEnsemble(
  Dict(
    :learners => [stacklearner,sk_learner,crt_learner,CT.PrunedTree(), CT.RandomForest(),CT.DecisionStumpAdaboost(),rf], 
    :output => :class,
  )
)
predict(votelearner)

# All learners are called in the same way.
@everywhere stackstacklearner = CT.StackEnsemble(
  Dict(
    :learners => [stacklearner,bestlearner,votelearner,sk_learner,crt_learner,CT.PrunedTree(), CT.RandomForest(),CT.DecisionStumpAdaboost(),rf], 
    #:keep_original_features => false,
    #:stacker_training_proportion => 0.3,
    :stacker => 
       CT.RandomForest(
          Dict(
             :output => :class,
             :impl_options => Dict(
               :num_subfeatures => nothing,
               :num_trees => 20,
               :partial_sampling => 0.7
             )
          )
       )
  )
)
predict(stackstacklearner)


acc=@parallel (vcat) for i=1:5
    res=predict(stacklearner)
    println(round(res))
    res
end
println("Acc=",round(mean(acc))," +/- ",round(std(acc)))

using DataFrames
function main(trials)
    learners=Dict(:votelearner=>votelearner,:bestlearner=>bestlearner,:stacklearner=>stacklearner,:stackstacklearner=>stackstacklearner)
    models=collect(keys(learners))
    ctable=@parallel (vcat) for model in models
        acc=@parallel (vcat) for i=1:trials
            res=predict(learners[model])
            println(model," => ",round(res))
            res
        end
        [model round(mean(acc)) round(std(acc)) length(acc)]
    end
    sorted=sortrows(ctable,by=(x)->x[2],rev=true) |> DataFrame
    rename!(sorted,Dict(:x1=>:model,:x2=>:mean_acc,:x3=>:std_acc,:x4=>:trials))
    return sorted
end

res=main(5) 

#@everywhere macro suppress_err(block)
#    quote
#        if ccall(:jl_generating_output, Cint, ()) == 0
#            ORIGINAL_STDERR = STDERR
#            err_rd, err_wr = redirect_stderr()
#            err_reader = @async readstring(err_rd)
#            value = $(esc(block))
#            @async wait(err_reader)
#            REDIRECTED_STDERR = STDERR
#            err_stream = redirect_stderr(ORIGINAL_STDERR)
#            return value
#        end
#    end
#end
