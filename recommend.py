import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset from CSV
data_path = r"C:\Users\12\Documents\Career Project fullfledg\course titles.csv"  # Replace this with the path to your CSV file
df = pd.read_csv(data_path)

# extract course titles and skills column from the dataframe 
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

def map_user_input_to_skills(user_input, skill_levels):
    """Map user input to skill levels."""
    user_skills = {}
    for skill, levels in skill_levels.items():
        for level in levels:
            if re.search(level, user_input, re.IGNORECASE):
                user_skills[skill] = level
    return user_skills

def identify_skill_gaps(current_skills, goals, skill_levels):
    """Identify skill gaps between current skills and goals."""
    skill_gaps = {}
    for skill, goal_level in goals.items():
        current_level = current_skills.get(skill, 'beginner')
        if skill_levels[skill].index(current_level) < skill_levels[skill].index(goal_level):
            skill_gaps[skill] = (current_level, goal_level)
    return skill_gaps

def recommend_resources_based_on_gaps(skill_gaps, top_n=3):
    """Recommend resources based on skill gaps."""
    recommended_resources = []
    for skill, (current_level, goal_level) in skill_gaps.items():
        query = f'{skill} {current_level} to {goal_level}'
        query_vector = vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, course_vectors).flatten()
        top_n_indices = cosine_similarities.argsort()[::-1][:top_n]
        recommended_resources.extend(df.iloc[top_n_indices][['course_title', 'url']].values.tolist())
    return recommended_resources

# Example Usage:
user_input = "I want to improve my Python skills to an advanced level."
learning_goals = {"Python": "advanced"}
skill_levels = {
    'Python': ['beginner', 'intermediate', 'advanced']
}

user_skills = map_user_input_to_skills(user_input, skill_levels)
skill_gaps = identify_skill_gaps(user_skills, learning_goals, skill_levels)
recommendations = recommend_resources_based_on_gaps(skill_gaps)
print(recommendations)
