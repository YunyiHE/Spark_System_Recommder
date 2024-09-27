# Recommend movies for user IDs 10 and 14
users = [10, 14]
top_recommendations = best_model.recommendForUsers(users, 15)

for user_id in users:
    user_recommendations = top_recommendations.filter(col("userId") == user_id).select("recommendations.movieId")
    print(f"Top 15 movie recommendations for user {user_id}:")
    user_recommendations.show(truncate=False)