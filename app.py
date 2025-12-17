import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from flask import Flask, request, render_template

app = Flask(__name__)

with open('bernoulli_naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)
    print(model)

with open('tfidf_vectorizer.pkl', 'rb') as f:
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
    lemmatize = WordNetLemmatizer()
    leammatized_words = []
    for j in filter_without_stopwords:
        leammatized_words.append(lemmatize.lemmatize(j))

    # joining back 
    joined_word = " ".join(leammatized_words)

    return joined_word

@app.route("/", methods = ['GET', 'POST'])
def home():
    msg = None
    if request.method == 'POST':

        review = request.form['review']
        label = ""

        clean = preprocessing(review)
        vectorize = vectorizer.transform([clean]).toarray()
        prediction = int(model.predict(vectorize)[0])
        confidence = model.predict_proba(vectorize)[0, 1]
        confidence_percentage = round(float(confidence * 100))

        if prediction == 1:
            label = "Customer is Positive :) about his/her review !"
        elif prediction == 0:
            label = "Customer is Negative :( about his/her review !"

        msg = {
            'review': review,
            'prediction': prediction,
            'label': label,
            'confidence': confidence_percentage,
            'model_info': str(model)
        }

    return render_template('index.html', msg=msg)

if __name__ == "__main__":
    app.run(debug=True)