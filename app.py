# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import pandas as pd  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai  
openai.api_key = 'YOUR API KEY HERE'
                                 
# Initialize Flask app
app = Flask(__name__) 

# Load dataset from CSV
data_path = r"C:\Users\12\Documents\Career Project fullfledg\course titles.csv"  # Replace this with the path to your CSV file
df = pd.read_csv(data_path)     
                                 
# Prepare course titles and skills 
course_titles = df['course_title']
skills = df['skills']

# Combine course titles and skills into a single column for TF-IDF vectorization
combined_descriptions = course_titles + ' ' + skills.apply(lambda x: ' '.join(eval(x)))

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed

# Fit the vectorizer to the combined descriptions
vectorizer.fit(combined_descriptions)

# Transform course descriptions into TF-IDF vectors
course_vectors = vectorizer.transform(combined_descriptions)

# Define function to recommend resources based on user input

def recommend_resources(user_skills, learning_goals):
    
    user_input = user_skills + ' ' + learning_goals
    user_vector = vectorizer.transform([user_input])

    similarities = cosine_similarity(user_vector, course_vectors)

    sorted_indices = similarities.argsort(axis=1)[0, ::-1]

    top_recommendations = df.iloc[sorted_indices[:3]]

    return top_recommendations[['course_title', 'url']].values.tolist()

# Define routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_skills = request.form["user_skills"]
        learning_goals = request.form["learning_goals"]
        recommendations = recommend_resources(user_skills, learning_goals)
        # return render_template("recommendation.html", recommendations=recommendations)
        
        # Generate a custom AI response
        prompt = (f"act as a professtionalist Given that the user has skills in {user_skills} and wants to learn {learning_goals}, "
                  "please provide a detailed explanation or guidance on how these courses will help in achieving their learning goals. "
                  "Create a detailed roadmap with bullet points, showing the estimated time for each learning step in brackets. "
                  "Provide the best resources those are helpful in the specific subject"
                  "Make the roadmap visually appealing and easy to follow.")
        
        response = openai.Completion.create(  
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=60
        )
       
        ai_response = response.choices[0].text.strip()
        
        return render_template("recommendation.html", recommendations=recommendations, 
                               user_skills=user_skills, 
                               learning_goals=learning_goals,
                               ai_response=ai_response)
    else:
        return render_template("index.html")
    

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
