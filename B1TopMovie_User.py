
from pyspark.sql import functions as F

df = spark.read.csv("/Users/User1/Desktop/MIE1628_2023_CloudComputing/A2/movies.csv", header=True)

# Calculate the total number of rows in the dataset
total_rows = df.count()
print("Total number of rows:", total_rows)

# Get the names of the headers
headers = df.columns
print("Header names:", headers)

# Count the distinct values for 'movieId,' 'userId,' and 'rating'
distinct_movie_ids = df.select('movieId').distinct().count()
distinct_user_ids = df.select('userId').distinct().count()
distinct_ratings = df.select('rating').distinct().count()

print("Distinct movieId count:", distinct_movie_ids)
print("Distinct userId count:", distinct_user_ids)
print("Distinct rating count:", distinct_ratings)



# Group by 'movieId' and calculate the average rating for each movie
average_ratings = df.groupBy('movieId').agg(F.avg('rating').alias('avg_rating'))

# Sort the results in descending order
sorted_ratings = average_ratings.sort(F.desc('avg_rating'))

# Select the top 5 movies with the highest ratings
top_20_movies = sorted_ratings.limit(20)

# Show the results
top_20_movies.show()

# Convert the 'rating' column to a numeric type
df = df.withColumn("rating", df["rating"].cast("double"))

# Group by 'userId' and calculate the average rating for each user
average_ratings = df.groupBy('userId').agg(F.avg('rating').alias('avg_rating'))

# Sort the results in descending order
sorted_ratings = average_ratings.sort(F.desc('avg_rating'))

# Select the top 15 users with the highest average ratings
top_15_users = sorted_ratings.limit(15)

# Show the results
top_15_users.show()


