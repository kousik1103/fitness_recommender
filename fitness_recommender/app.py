from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv(r"C:\Users\kousi\OneDrive\Desktop\fitness_recommender\Personalized_Fitness_Recommender_Dataset.csv")

# Initialize Flask app
app = Flask(__name__)

# Preprocess data
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[["FitnessGoal", "ActivityPreference", "DietPreference"]]).toarray()
df["EncodedFeatures"] = list(encoded_features)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    # Get user input
    age = int(request.form["age"])
    gender = request.form["gender"]
    fitness_goal = request.form["fitness_goal"]
    activity_preference = request.form["activity_preference"]
    diet_preference = request.form["diet_preference"]

    # Encode user input
    user_features = encoder.transform([[fitness_goal, activity_preference, diet_preference]]).toarray()

    # Compute similarity
    similarities = cosine_similarity([user_features[0]], encoded_features)
    similar_index = similarities.argmax()

    # Get recommendation
    recommended_workout = df.iloc[similar_index]["RecommendedWorkout"]
    recommended_diet = df.iloc[similar_index]["RecommendedDiet"]

    return render_template(
        "result.html",
        workout=recommended_workout,
        diet=recommended_diet
    )

if __name__ == "__main__":
    app.run(debug=True)
