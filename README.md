Copyright for portions of project CombineML.jl are held by Samuel Jenkins, 2014
as part of project Orchestra.jl. All other copyright for project CombineML.jl
are held by Paulito Palmes, 2016.

The CombineML.jl package is licensed under the MIT "Expat" License:

You may also be interested to [TSML (Time Series Machine Learning)](https://github.com/IBM/TSML.jl) package.

# CombineML

[![Join the chat at https://gitter.im/CombineML-jl/Lobby](https://badges.gitter.im/CombineML-jl/Lobby.svg)](https://gitter.im/CombineML-jl/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Julia 1.0/Linux/OSX: [![Build Status](https://travis-ci.org/ppalmes/CombineML.jl.svg?branch=master)](https://travis-ci.org/ppalmes/CombineML.jl)
[![Coverage Status](https://coveralls.io/repos/github/ppalmes/CombineML.jl/badge.svg?branch=master)](https://coveralls.io/github/ppalmes/CombineML.jl?branch=master)

CombineML is a heterogeneous ensemble learning package for the Julia programming
language. It is driven by a uniform machine learner API designed for learner
composition.

## Getting Started

See the notebook for demo: [MetaModeling.ipnyb](https://github.com/ppalmes/CombineML.jl/blob/master/MetaModeling.ipynb)

We will cover how to predict on a dataset using CombineML.

### Obtain Data

A tabular dataset will be used to obtain our features and labels. 

This will be split it into a training and test set using holdout method.

```julia
import CombineML
using CombineML.Util
using CombineML.Transformers

try
  import RDatasets
catch
  using Pkg
  Pkg.add("RDatasets")
  import RDatasets
end

# use shorter module names
CU=CombineML.Util
CT=CombineML.Transformers
RD=RDatasets

# Obtain features and labels
dataset = RD.dataset("datasets", "iris")
features = convert(Matrix,dataset[:, 1:(end-1)])
labels = convert(Array,dataset[:, end])

# Split into training and test sets
(train_ind, test_ind) = CU.holdout(size(features, 1), 0.3)
```

### Create a Learner

A transformer processes features in some form. Coincidentally, a learner is a subtype of transformer.

A transformer can be created by instantiating it, taking an options dictionary as an optional argument. 

All transformers, including learners are called in the same way.

```julia
# Learner with default settings
learner = CT.PrunedTree()

# Learner with some of the default settings overriden
learner = CT.PrunedTree(Dict(
  :impl_options => Dict(
    :purity_threshold => 1.0
  )
))

# All learners are called in the same way.
learner = CT.StackEnsemble(Dict(
  :learners => [
    CT.PrunedTree(), 
    CT.RandomForest(),
    CT.DecisionStumpAdaboost()
  ], 
  :stacker => CT.RandomForest()
))
```

### Create a Pipeline

Normally we may require the use of data pre-processing before the features are passed to the learner.

We shall use a pipeline transformer to chain many transformers in sequence.

In this case we shall one hot encode categorical features, impute NA values and numerically standardize before we call the learner.

```julia
# Create pipeline
pipeline = CT.Pipeline(Dict(
  :transformers => [
    CT.OneHotEncoder(), # Encodes nominal features into numeric
    CT.Imputer(), # Imputes NA values
    CT.StandardScaler(), # Standardizes features 
    learner # Predicts labels on features
  ]
))
```

### Train and Predict

Training is done via the `fit!` function, predicton via `transform!`. 

All transformers, provide these two functions. They are always called the same way.

```julia
# Train
CT.fit!(pipeline, features[train_ind, :], labels[train_ind])

# Predict
predictions = CT.transform!(pipeline, features[test_ind, :])
```

### Assess

Finally we assess how well our learner performed.

```julia
# Assess predictions
result = CU.score(:accuracy, labels[test_ind], predictions)
```

## Available Transformers

Outlined are all the transformers currently available via CombineML.

### CombineML

#### Baseline (CombineML.jl Learner)

Baseline learner that by default assigns the most frequent label.
```julia

try
  import StatsBase
catch
  using Pkg
  Pkg.add("StatsBase")
  import StatsBase
end

learner = CT.Baseline(Dict(
  # Output to train against
  # (:class).
  :output => :class,
  # Label assignment strategy.
  # Function that takes a label vector and returns the required output.
  :strategy => StatsBase.mode
))
```

#### Identity (CombineML.jl Transformer)

Identity transformer passes the features as is.
```julia
transformer = CT.Identity()
```

#### VoteEnsemble (CombineML.jl Learner)

Set of machine learners that majority vote to decide prediction.
```julia
learner = CT.VoteEnsemble(Dict(
  # Output to train against
  # (:class).
  :output => :class,
  # Learners in voting committee.
  :learners => [CT.PrunedTree(), CT.DecisionStumpAdaboost(), CT.RandomForest()]
))
```

#### StackEnsemble (CombineML.jl Learner)

Ensemble where a 'stack' learner learns on a set of learners' predictions.
```julia
learner = CT.StackEnsemble(Dict(
  # Output to train against
  # (:class).
  :output => :class,
  # Set of learners that produce feature space for stacker.
  :learners => [CT.PrunedTree(), CT.DecisionStumpAdaboost(), CT.RandomForest()],
  # Machine learner that trains on set of learners' outputs.
  :stacker => CT.RandomForest(),
  # Proportion of training set left to train stacker itself.
  :stacker_training_proportion => 0.3,
  # Provide original features on top of learner outputs to stacker.
  :keep_original_features => false
))
```

#### BestLearner (CombineML.jl Learner)

Selects best learner out of set. 
Will perform a grid search on learners if options grid is provided.
```julia
learner = CT.BestLearner(Dict(
  # Output to train against
  # (:class).
  :output => :class,
  # Function to return partitions of instance indices.
  :partition_generator => (features, labels) -> kfold(size(features, 1), 5),
  # Function that selects the best learner by index.
  # Arg learner_partition_scores is a (learner, partition) score matrix.
  :selection_function => (learner_partition_scores) -> findmax(mean(learner_partition_scores, 2))[2],      
  # Score type returned by score() using respective output.
  :score_type => Real,
  # Candidate learners.
  :learners => [CT.PrunedTree(), CT.DecisionStumpAdaboost(), CT.RandomForest()],
  # Options grid for learners, to search through by BestLearner.
  # Format is [learner_1_options, learner_2_options, ...]
  # where learner_options is same as a learner's options but
  # with a list of values instead of scalar.
  :learner_options_grid => nothing
))
```

#### OneHotEncoder (CombineML.jl Transformer)

Transforms nominal features into one-hot form 
and coerces the instance matrix to be of element type Float64.
```julia
transformer = CT.OneHotEncoder(Dict(
  # Nominal columns
  :nominal_columns => nothing,
  # Nominal column values map. Key is column index, value is list of
  # possible values for that column.
  :nominal_column_values_map => nothing
))
```

#### Imputer (CombineML.jl Transformer)

Imputes NaN values from Float64 features.
```julia
transformer = CT.Imputer(Dict(
  # Imputation strategy.
  # Statistic that takes a vector such as mean or median.
  :strategy => mean
))
```

#### Pipeline (CombineML.jl Transformer)

Chains multiple transformers in sequence.
```julia
transformer = CT.Pipeline(Dict(
  # Transformers as list to chain in sequence.
  :transformers => [CT.OneHotEncoder(), CT.Imputer()],
  # Transformer options as list applied to same index transformer.
  :transformer_options => nothing
))
```

#### Wrapper (CombineML.jl Transformer)

Wraps around an CombineML transformer.
```julia
transformer = Wrapper(Dict(
  # Transformer to call.
  :transformer => CT.OneHotEncoder(),
  # Transformer options.
  :transformer_options => nothing
))
```


### Julia

#### PrunedTree (DecisionTree.jl Learner)

Pruned CART decision tree.
```julia
learner = CT.PrunedTree(Dict(
  # Output to train against
  # (:class).
  :output => :class,
  # Options specific to this implementation.
  :impl_options => Dict(
    # Merge leaves having >= purity_threshold combined purity.
    :purity_threshold => 1.0,
    # Maximum depth of the decision tree (default: no maximum).
    :max_depth => -1,
    # Minimum number of samples each leaf needs to have.
    :min_samples_leaf => 1,
    # Minimum number of samples in needed for a split.
    :min_samples_split => 2,
    # Minimum purity increase needed for a split.
    :min_purity_increase => 0.0
  ) 
))
```

#### RandomForest (DecisionTree.jl Learner)

Random forest (CART).
```julia
learner = CT.RandomForest(Dict(
  # Output to train against
  # (:class).
  :output => :class,
  # Options specific to this implementation.
  :impl_options => Dict(
    # Number of features to train on with trees (default: 0, keep all).
    # Good values are square root or log2 of total number of features, rounded.
    # Number of trees in forest.
    :num_trees => 10,
    # Proportion of trainingset to be used for trees.
    :partial_sampling => 0.7,
    # Maximum depth of each decision tree (default: no maximum).
    :max_depth => -1
  )
))
```

#### DecisionStumpAdaboost (DecisionTree.jl Learner)

Adaboosted decision stumps.
```julia
learner = CT.DecisionStumpAdaboost(Dict(
  # Output to train against
  # (:class).
  :output => :class,
  # Options specific to this implementation.
  :impl_options => Dict(
    # Number of boosting iterations.
    :num_iterations => 7
  )
))
```

#### PCA (DimensionalityReduction.jl Transformer)

Principal Component Analysis rotation
on features.
Features ordered by maximal variance descending.

Fails if zero-variance feature exists. Based on MultivariateStats PCA
```julia
transformer = CT.PCA(Dict(
  :pratio => 1.0,
  :maxoutdim => 5
))
```

#### StandardScaler (MLBase.jl Transformer)

Standardizes each feature using (X - mean) / stddev.
Will produce NaN if standard deviation is zero.
```julia
transformer = CT.StandardScaler(Dict(
  # Center features
  :center => true,
  # Scale features
  :scale => true
))
```

### Python

See the scikit-learn [API](http://scikit-learn.org/stable/modules/classes.html) for what options are available per learner.

#### SKLLearner (scikit-learn 0.15 Learner)

Wrapper for scikit-learn that provides access to most learners.

Options for the specific scikit-learn learner is to be passed
in `options[:impl_options]` dictionary.

Available Classifiers:

  - AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, LDA, LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV, SGDClassifier, KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid, QDA, SVC, LinearSVC, NuSVC, DecisionTreeClassifier, GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
  
Available Regressors:
- SVR, Ridge, RidgeCV, Lasso, ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor, PassiveAggressiveRegressor, KernelRidge, KNeighborsRegressor, RadiusNeighborsRegressor, GaussianProcessRegressor, DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, IsotonicRegression, MLPRegressor


```julia
# classifier example
learner = CT.SKLLearner(Dict(
  # Output to train against
  # (classification).
  :output => :class,
  :learner => "LinearSVC",
  # Options specific to this implementation.
  :impl_options => Dict()
))

# regression example
learner = CT.SKLLearner(Dict(
  # Output to train against
  # (regression).
  :output => :reg,
  :learner => "GradientBoostingRegressor",
  # Options specific to this implementation.
  :impl_options => Dict()
))
```


### R

RCall is used to interface with caret learners.

R 'caret' library offers more than 100 learners. 
See [here](http://caret.r-forge.r-project.org/modelList.html) for more details.

#### CRTLearner (caret 6.0 Learner)

CARET wrapper that provides access to all learners.

Options for the specific CARET learner is to be passed
in `options[:impl_options]` dictionary.
```julia
learner = CT.CRTLearner(Dict(
  # Output to train against
  # (:class).
  :output => :class,
  :learner => "svmLinear",
  :impl_options => Dict()
))
```


## Known Limitations

Learners have only been tested on numeric features. 

Inconsistencies may result in using nominal features directly without a numeric transformation (i.e. OneHotEncoder).

## Misc

The links provided below will only work if you are viewing this in the GitHub repository.

### Changes

See [CHANGELOG.yml](CHANGELOG.yml).

### License

MIT "Expat" License. See [LICENSE.md](LICENSE.md).
