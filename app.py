import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, request, render_template

app = Flask(__name__)

with open('multinomial_naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)
    print(model)

with open('count_vectorize_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    print(vectorizer)

def preprocessing(random_input):

    # lower case conversion
    random_input.lower()

    # clean uniques
    regex_unique = re.compile('[^\w\s]')
    clean_unique = re.sub(regex_unique, " ", random_input).strip()

    # clean space
    clean_space = re.sub(r"\s+", " ", clean_unique).strip()

    # clean stop word
    stop_words = stopwords.words('english')
    filter_without_stopwords = []
    for i in clean_space.split():
        if i not in stop_words:
            filter_without_stopwords.append(i)

    # stemming 
    p_stemmer = PorterStemmer()
    stemmed_words = []
    for j in filter_without_stopwords:
        stemmed_words.append(p_stemmer.stem(j))
        
    # joining back 
    joined_word = " ".join(stemmed_words)

    return joined_word

@app.route("/", methods = ['GET', 'POST'])
def home():
    msg = None
    if request.method == 'POST':
        review = request.form['review']
        clean = preprocessing(review)
        vectorize = vectorizer.transform([clean]).toarray()
        prediction = model.predict(vectorize)[0]
        msg = f"You entered: {prediction}"
    return render_template('index.html', msg=msg)

if __name__ == "__main__":
    app.run(debug=True)