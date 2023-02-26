from flask import Flask, request, render_template
import pandas as pd
import joblib
import re
import numpy as np
import nltk
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

app.static_folder = './front/static'
app.template_folder = './front/templates'

# Load the machine learning model
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # get user inputs from the form
        title = request.form['title']
        genres = request.form.getlist('genre[]')
        description = request.form['description']
        anime_type = request.form['type']
        producer = request.form.getlist('producer[]')
        studio = request.form.getlist('studio[]')

        # preprocess data 
        df = pd.DataFrame({"Title": [title],'Genres': [','.join(genres)], 'Description': [description], 'Type': [anime_type], 'Producer': [','.join(producer)], 'Studio': [','.join(studio)]})
        df_preprocess = preprocess_data(df)

        # Use the model to make a prediction
        prediction = model.predict(df_preprocess)
        #prediction = 6,35

        return render_template('rating.html',title=title, genres=genres, description=description, anime_type=anime_type, producer=producer, studio=studio, prediction=prediction)

    # If the request method is GET, render the home template
    return render_template('home.html')

def preprocess_text(text):
    # Remove ' \r\n \r\n' plus last sentence
    text = re.sub(r'\s*(\r\n\s*)+|\[Written by .*?\]+|\(Source: .*?\)\s*', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word not in nltk.corpus.stopwords.words('english')]
    return words

def get_vector(text):
    tokens = [token for token in text]
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return sum(vectors)/len(vectors)

def get_title_vector(title):
    words = title.lower().split()
    vectors = [model.wv[word] for word in words if word in model.wv.key_to_index]
    if len(vectors) > 0:
        return sum(vectors) / len(vectors)
    else:
        return np.zeros(model.vector_size)
    
def preprocess_data(df):
    # preprocess Genres
    df_genre = df[['Title','Genre']]
    df_genre['Genre'] = df_genre['Genre'].apply(lambda x: "['NoGenre']" if pd.isna(x) else x)
    df_genre['Genre'] = df_genre['Genre'].apply(lambda x: eval(x))
    df_genre['Genre'] = df_genre['Genre'].apply(lambda x: '|'.join(x))
    genres = df_genre['Genre'].str.get_dummies('|')
    df_preprocess = pd.concat([df_preprocess, genres], axis=1).drop('Genre', axis=1)

    # preprocess Producer
    df_producer = df[['Title','Producer']]
    df_producer['Producer'] = df_producer['Producer'].apply(lambda x: "['NoProducer']" if pd.isna(x) else x)
    df_producer['Producer'] = df_producer['Producer'].apply(lambda x: eval(x))
    df_producer['Producer'] = df_producer['Producer'].apply(lambda x: '|'.join(x))
    producers = df_producer['Producer'].str.get_dummies('|')
    df_preprocess = pd.concat([df_preprocess, producers], axis=1).drop('Producer', axis=1)

    # preprocess Studio
    df_studio = df[['Title','Studio']]
    df_studio['Studio'] = df_studio['Studio'].apply(lambda x: "['NoStudio']" if pd.isna(x) else x)
    df_studio['Studio'] = df_studio['Studio'].apply(lambda x: eval(x))
    df_studio['Studio'] = df_studio['Studio'].apply(lambda x: '|'.join(x))
    studio = df_studio['Studio'].str.get_dummies('|')
    studio
    df_preprocess = pd.concat([df_preprocess, studio], axis=1).drop('Studio', axis=1)

    # preprocess Type
    df_preprocess = pd.get_dummies(df_preprocess, columns = ["Type"])

    # preprocess Synopsis
    df_synospsis = df_preprocess[['Title', 'Synopsis']]
    df_synospsis['Synopsis'] = df_synospsis['Synopsis'].fillna('NoSynopsis')
    df_synospsis['Synopsis'] = df_synospsis['Synopsis'].apply(preprocess_text)

    Word2Vec(df_synospsis['Synopsis'], vector_size=100, window=5, min_count=1, workers=4)
    df_synospsis['vector_synopsis'] = df_synospsis['Synopsis'].apply(get_vector)
    df_preprocess['Synopsis'] = df_synospsis['vector_synopsis']
    df_preprocess['Synopsis'] = df_preprocess['Synopsis'].apply(lambda x : x.tolist())

    # preprocess Title
    Word2Vec(df_preprocess['Title'].apply(preprocess_text), vector_size=100, window=5, min_count=1, workers=4)
    df_preprocess['Title'] = df_preprocess['Title'].apply(get_title_vector)
    scaler = MinMaxScaler()
    df_preprocess['Title'] = scaler.fit_transform(df_preprocess['Title'].tolist())

    # final preprocess for prediction
    #vectorizer = TfidfVectorizer(stop_words='english')
    #synopsis_matrix = vectorizer.fit_transform(df_preprocess['Synopsis'])
    #similarity_matrix = cosine_similarity(synopsis_matrix)

    #similar_movies_indices = {}
    #for i, row in df.iterrows():
    #    # remove the anime for its own list
    #    similar_indices = similarity_matrix[i].argsort()[::-1][1:]
    #    similar_movies_indices[i] = similar_indices

    #predicted_ratings = []
    #for i, row in df.iterrows():
    #    similar_indices = similar_movies_indices[i]
    #    similar_ratings = df.iloc[similar_indices]['Rating']
    #    predicted_rating = similar_ratings.mean()
    #    predicted_ratings.append(predicted_rating)
    #df['Predicted_Rating'] = predicted_ratings
    
    return df_preprocess

if __name__ == '__main__':
    app.run(debug=True)