
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

# Create a ParamGridBuilder
param_grid = ParamGridBuilder() \
    .addGrid(als.rank, [10, 20, 30]) \
    .addGrid(als.maxIter, [5, 10, 15]) \
    .addGrid(als.regParam, [0.01, 0.1, 0.2]) \
    .addGrid(als.implicitPrefs, [False, True]) \
    .addGrid(als.alpha, [0.01, 0.1, 0.2]) \
    .addGrid(als.nonnegative, [False, True]) \
    .build()

# Create a CrossValidator
crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=5)  # can adjust the number of folds

# Fit the CrossValidator to the training data
cv_model = crossval.fit(train_data)

# Get the best model from cross-validation
best_model = cv_model.bestModel

# Get the best hyperparameters
best_rank = best_model.rank
best_max_iter = best_model._java_obj.parent().getMaxIter()
best_reg_param = best_model._java_obj.parent().getRegParam()
best_implicit_prefs = best_model._java_obj.parent().getImplicitPrefs()
best_alpha = best_model._java_obj.parent().getAlpha()
best_nonnegative = best_model._java_obj.parent().getNonnegative()

# Make predictions using the best model
best_predictions = best_model.transform(test_data)

# Evaluate the best model
best_rmse = evaluator.evaluate(best_predictions)
print(f"Best Root Mean Squared Error (RMSE): {best_rmse}")

# Print the best parameters
print(f"Best rank: {best_rank}")
print(f"Best maxIter: {best_max_iter}")
print(f"Best regParam: {best_reg_param}")
print(f"Best implicitPrefs: {best_implicit_prefs}")
print(f"Best alpha: {best_alpha}")
print(f"Best nonnegative: {best_nonnegative}")


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create a ParamGridBuilder
param_grid = ParamGridBuilder() \
    .addGrid(als.rank, [10, 20, 30]) \
    .addGrid(als.maxIter, [5, 10, 15]) \
    .addGrid(als.regParam, [0.01, 0.1, 0.2]) \
    .build()

# Create a CrossValidator
crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=5)  # can adjust the number of folds

# Fit the CrossValidator to the training data
cv_model = crossval.fit(train_data)

# Get the best model from cross-validation
best_rank = cv_model.best_model.rank
best_max_iter = cv_model.bestModel._java_obj.parent().getMaxIter()
best_reg_param = cv_model.bestModel._java_obj.parent().getRegParam()
# Make predictions using the best model
best_predictions = best_model.transform(test_data)

# Evaluate the best model
best_rmse = evaluator.evaluate(best_predictions)
print(f"Best Root Mean Squared Error (RMSE): {best_rmse}")

# Print the best parameters
print(f"Best rank: {best_model.rank}")
print(f"Best maxIter: {best_max_iter}")
print(f"Best regParam: {best_reg_param}")