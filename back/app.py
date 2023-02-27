from flask import Flask, request, render_template
import pandas as pd
import joblib
import re
import numpy as np
import nltk
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

app.static_folder = '../front/static'
app.template_folder = '../front/templates'

# Load the machine learning model
#model = joblib.load('./model.pkl')
model = joblib.load('/app/model.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
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
        df = pd.DataFrame({"Title": [title],'Genres': str(genres), 'Synopsis': [description], 'Type': [anime_type], 'Producer': str(producer), 'Studio': str(studio)})
        print(df)
        
        df_preprocess = preprocess_data(df)
        df_preprocess = df_preprocess.drop("Synopsis", axis = 1)

        # create an empty dataset with the corresponding columns
        with open('./cols.txt', 'r') as f:
            cols = f.read()

        cols=eval(cols)
        df_final = pd.DataFrame(columns=cols)
        df_final.loc[0, cols] = 0

        df_final.loc[0, 'Title'] = df_preprocess['Title'][0]
        matching_elements = [elem for elem in cols if any(val in elem for val in genres)]
        matching_elements.append(elem for elem in cols if any(val in elem for val in producer))
        matching_elements.append(elem for elem in cols if any(val in elem for val in studio))
        matching_elements.append(elem for elem in cols if any(val in elem for val in anime_type))

        df_final.loc[0, matching_elements] = 1
        df_final = df_final.iloc[:,:-4]

        # Use the model to make a prediction
        prediction = model.predict(df_final)

        return render_template('rating.html',title=title, genres=genres, description=description, anime_type=anime_type, producer=producer, studio=studio, prediction=prediction)

    # If the request method is GET, render the home template
    


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

def get_vector(text, model):
    tokens = [token for token in text]
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return sum(vectors)/len(vectors)

def get_title_vector(title, model):
    words = title.lower().split()
    vectors = [model.wv[word] for word in words if word in model.wv.key_to_index]
    if len(vectors) > 0:
        return sum(vectors) / len(vectors)
    else:
        return np.zeros(model.vector_size)
    
def preprocess_data(df):

    # preprocess Genres
    df_genre = df[['Title','Genres']]
    df_genre['Genres'] = df_genre['Genres'].apply(lambda x: "['NoGenre']" if pd.isna(x) else x)
    df_genre['Genres'] = df_genre['Genres'].apply(lambda x: eval(x))
    df_genre['Genres'] = df_genre['Genres'].apply(lambda x: '|'.join(x))
    genres = df_genre['Genres'].str.get_dummies('|')
    df = pd.concat([df, genres], axis=1).drop('Genres', axis=1)

    # preprocess Producer
    df_producer = df[['Title','Producer']]
    df_producer['Producer'] = df_producer['Producer'].apply(lambda x: "['NoProducer']" if pd.isna(x) else x)
    df_producer['Producer'] = df_producer['Producer'].apply(lambda x: eval(x))
    df_producer['Producer'] = df_producer['Producer'].apply(lambda x: '|'.join(x))
    producers = df_producer['Producer'].str.get_dummies('|')
    df = pd.concat([df, producers], axis=1).drop('Producer', axis=1)

    # preprocess Studio
    df_studio = df[['Title','Studio']]
    df_studio['Studio'] = df_studio['Studio'].apply(lambda x: "['NoStudio']" if pd.isna(x) else x)
    df_studio['Studio'] = df_studio['Studio'].apply(lambda x: eval(x))
    df_studio['Studio'] = df_studio['Studio'].apply(lambda x: '|'.join(x))
    studio = df_studio['Studio'].str.get_dummies('|')
    studio
    df = pd.concat([df, studio], axis=1).drop('Studio', axis=1)

    # preprocess Type
    df = pd.get_dummies(df, columns = ["Type"])

    # preprocess Synopsis
    df_synospsis = df[['Title', 'Synopsis']]
    df_synospsis['Synopsis'] = df_synospsis['Synopsis'].fillna('NoSynopsis')
    df_synospsis['Synopsis'] = df_synospsis['Synopsis'].apply(preprocess_text)

    model2 = Word2Vec(df_synospsis['Synopsis'], vector_size=100, window=5, min_count=1, workers=4)
    df_synospsis['vector_synopsis'] = df_synospsis['Synopsis'].apply(lambda x : get_vector(x,model2))
    df['Synopsis'] = df_synospsis['vector_synopsis']
    df['Synopsis'] = df['Synopsis'].apply(lambda x : x.tolist())

    # preprocess Title
    model3 = Word2Vec(df['Title'].apply(preprocess_text), vector_size=100, window=5, min_count=1, workers=4)
    df['Title'] = df['Title'].apply(lambda x : get_title_vector(x,model3))
    scaler = MinMaxScaler()
    df['Title'] = scaler.fit_transform(df['Title'].tolist())

    return df

if __name__ == '__main__':
    app.run(debug=True)