import json
import pickle
import requests
from flask_wtf import FlaskForm
from flask_session import Session
from wtforms.validators import DataRequired
from wtforms import TextAreaField, SubmitField
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request,redirect, url_for, session, flash


app = Flask(__name__, template_folder='./src/templates/',
            static_folder='./src/static/')


# Configurations
app.config['SECRET_KEY'] = 'sickrat'
app.config['SESSION_TYPE'] = 'filesystem' 

Session(app)

# Load the SVM model
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models', 'pipeline.pkl')
if 'pipeline' not in globals():
    with open(model_path, 'rb') as model_svm:
        pipeline = pickle.load(model_svm)
        print("Model loaded successfully")

# Load movie data from API
import requests_cache

requests_cache.install_cache('omdb_cache', expire_after=3600)  # Cache API responses for 1 hour

def load_movies(query):
    page_number = 3
    results = []
    all_results = []

    for page in range(1, page_number + 1):
        url = f"http://www.omdbapi.com/?apikey=e308b9e6&s={query}&page={page}"
        response = requests.get(url)
        print(f"OMDB API Response for page {page}: {response.json()}")
        data = response.json()
        if 'Search' in data:
            results.extend(data['Search'])
        else:
            break

    # Fetch plot for each movie
    for movie in results:
        plot_url = f"http://www.omdbapi.com/?apikey=e308b9e6&i={movie['imdbID']}&plot=full"
        plot_response = requests.get(plot_url)
        print(f"OMDB API Plot Response for {movie['Title']}: {plot_response.json()}")
        plot_data = plot_response.json()
        if 'Plot' in plot_data:
            all_results.append(plot_data)

    return all_results


def load_movie(title):
    url = f"http://www.omdbapi.com/?apikey=e308b9e6&t={title}"
    response = requests.get(url)
    if response:
        return [response.json()]
    else:
        return []

# Form class
class ReviewForm(FlaskForm):
    review = TextAreaField('Review', validators=[DataRequired()], render_kw={
                           "placeholder": "Enter your review here"})
    submit = SubmitField('Submit')


# Content based filtering
class ContentBasedFilter:
    def __init__(self, movies):
        self.movies = movies
        self.features = []
        self.imdb_ids = []
        self.vectorizer = TfidfVectorizer()
        self.similarities = None
        self._extract_features()
        self._calculate_similarity()

    def _extract_features(self):
        for movie in self.movies:
            feature = movie["Genre"] + " " + movie["Plot"] + " " + movie["Director"] + " " + movie["Actors"]
            print(f"Extracted feature for {movie['Title']}: {feature}")
            self.features.append(feature)
            self.imdb_ids.append(movie["imdbID"])

    def _calculate_similarity(self):
        if not hasattr(self, 'similarities'):
            X = self.vectorizer.fit_transform(self.features)
            self.similarities = cosine_similarity(X, X)
            print(f"Similarity matrix shape: {self.similarities.shape}")

    def recommend_similar_movies(self, imdb_id):
        movie_index = self.imdb_ids.index(imdb_id)
        similarity_scores = list(enumerate(self.similarities[movie_index]))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:]
        similar_movie_indices = [score[0] for score in similarity_scores[:10]]
        
        all_results = [self.movies[index] for index in similar_movie_indices]
        
        return all_results



@app.route('/', methods=['POST', 'GET'])
def index():
    print("Accessed index route")
    movies = []
    form = ReviewForm()
    title = 'Review and Recommend'
    
    if request.method == 'POST' and form.validate_on_submit():
        movie_id = request.form['movie_id']
        review = form.review.data
        print(f"Form submitted: Movie ID: {movie_id}, Review: {review}")
        prediction = pipeline.predict([review])
        if prediction == ['pos']:
            cbf = ContentBasedFilter(load_movies('all'))
            recommended_movies = cbf.recommend_similar_movies(movie_id)
            session['recommended_movies'] = recommended_movies
            flash('Thanks for the positive review, here are your movie recommendations!', 'info')
            return redirect(url_for('recommended_movies_page'))
    
        else:
            flash('Sorry to hear about that, you can keep on reviewing.', 'danger')
            return redirect(url_for('index'))

    genre = request.args.get('genre')
    search = request.args.get('search')
    if genre:
        movies = load_movies(genre)
        title = genre.capitalize()
    elif search:
        movies = load_movie(search)
        title = search.capitalize()
    else:
        movies = load_movies('all')
    return render_template('index.html', form=form, movies=movies, title=title)

@app.route('/recommended', methods=['POST','GET'])
def recommended_movies_page():
    print("Accessed recommended_movies_page route")
    recommended_movies = session.get('recommended_movies')
    title = 'Recommended Movies'
    return render_template('recommended.html', movies=recommended_movies, title=title)
 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5009)

from flask import Flask, request, jsonify
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)

# Load the pre-trained model and TF-IDF vectorizer
with open('models/model_svm.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/pipeline.pkl', 'rb') as pipeline_file:
    pipeline = pickle.load(pipeline_file)

# Initialize the TF-IDF vectorizer and compute the TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(pipeline['tfidf_matrix'])

# Compute the cosine similarities
similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

def preprocess_data():
    global tfidf_matrix, similarities
    tfidf_matrix = tfidf_vectorizer.fit_transform(pipeline['tfidf_matrix'])
    similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route('/recommend', methods=['POST'])
def recommend_similar_movies():
    data = request.get_json()
    movie_id = data.get('movie_id')
    review = data.get('review')

    if similarities is None:
        preprocess_data()

    # Get the index of the movie
    movie_index = pipeline['movie_id_to_index'][movie_id]

    # Get the cosine similarity scores for the movie
    similarity_scores = list(enumerate(similarities[movie_index]))

    # Sort the movies based on the similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the top 5 similar movies (excluding the movie itself)
    top_similar_movies = similarity_scores[1:6]

    # Get the movie IDs of the top similar movies
    recommended_movie_ids = [pipeline['index_to_movie_id'][i[0]] for i in top_similar_movies]

    return jsonify(recommended_movie_ids)

if __name__ == "__main__":
    app.run(debug=True)
