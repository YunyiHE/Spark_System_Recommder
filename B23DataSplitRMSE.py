
#70% training, 30% testing

df = spark.read.csv("/Users/User1/Desktop/MIE1628_2023_CloudComputing/A2/movies.csv", header=True)
# Cast all columns to IntegerType
# Cast all columns to IntegerType
df = df.withColumn("movieId", df["movieId"].cast("int"))
df = df.withColumn("rating", df["rating"].cast("int"))
df = df.withColumn("userId", df["userId"].cast("int"))
# Split the data into 70% training and 30% testing
train_data, test_data = df.randomSplit([0.7, 0.3])


from pyspark.ml.recommendation import ALS

# Build the ALS model
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model70 = als.fit(train_data)

# Make predictions on the test data
predictions = model70.transform(test_data)
#predictions.collect()

from pyspark.ml.evaluation import RegressionEvaluator

# Evaluate the model
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")

#80% training, 20% testing

df = spark.read.csv("/Users/User1/Desktop/MIE1628_2023_CloudComputing/A2/movies.csv", header=True)
# Cast all columns to IntegerType
# Cast all columns to IntegerType
df = df.withColumn("movieId", df["movieId"].cast("int"))
df = df.withColumn("rating", df["rating"].cast("int"))
df = df.withColumn("userId", df["userId"].cast("int"))
train_data, test_data = df.randomSplit([0.8, 0.2])


from pyspark.ml.recommendation import ALS


# Build the ALS model
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
model80 = als.fit(train_data)

# Make predictions on the test data
predictions = model80.transform(test_data)
#predictions.collect()

from pyspark.ml.evaluation import RegressionEvaluator

# Evaluate the model
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")



