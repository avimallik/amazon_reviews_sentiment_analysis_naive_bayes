# Amazon Reviews Sentiment Analysis using Naive Bayes

This project performs sentiment analysis on Amazon reviews to classify them as either **Positive** or **Negative**. It utilizes Natural Language Processing (NLP) techniques for text cleaning and the **Naive Bayes** algorithm, such as **Gaussian**, **Multinmial**, and **Bernoulli** for classification, deployed via a **Flask** web application.

## Project Workflow

### 1. Data Preprocessing
The dataset is loaded and basic exploratory data analysis is performed, including checking for null values and visualizing the class distribution (Positive vs. Negative).
* **Lower Case Conversion**: All review text is converted to lowercase to ensure consistency during analysis.

### 2. Clean Unique Words
A regex-based cleaning function is applied to remove special characters and punctuation while preserving alphanumeric characters and spaces.

### 3. Clean Stop Words
Common English words (stop words) that do not contribute to sentiment are removed using the NLTK library.

### 4. Lemmatization
The `WordNetLemmatizer` is used to reduce words to their base or root form (e.g., "running" becomes "run"), which helps in reducing the dimensionality of the feature space.

### 5. TF-IDF Vectorization
The cleaned text is converted into numerical features using `TfidfVectorizer` with a maximum of 4,000 features and a n-gram range of (1, 2).

### 6. Model Evaluation
Three variations of Naive Bayes were tested: **Gaussian**, **Multinomial**, and **Bernoulli**.
* **Bernoulli Naive Bayes** achieved the highest accuracy, approximately **92.9%** on the test set.

### 7. Confusion Matrix
Confusion matrices were generated for each model to visualize performance, showing the true vs. predicted counts for both classes.

**Bernoulli Naive Bayes Results:**
![Bernoulli NB](https://github.com/avimallik/amazon_reviews_sentiment_analysis_naive_bayes/blob/master/confusion_matrix_gaussian_nb.png?raw=true)

**Bernoulli Naive Bayes Results:**
![Bernoulli NB](https://github.com/avimallik/amazon_reviews_sentiment_analysis_naive_bayes/blob/master/confusion_matrix_bernoulli_nb.png?raw=true)

**Multinomial Naive Bayes Results:**
![Multinomial NB](https://github.com/avimallik/amazon_reviews_sentiment_analysis_naive_bayes/blob/master/confusion_matrix_multinomial_nb.png?raw=true)

### 8. Classification Report
A detailed classification report provides metrics such as **Precision**, **Recall**, and **F1-score**, highlighting the model's high performance in identifying positive reviews.

### 9. Inference
A dedicated inference function processes raw input text through the same cleaning and vectorization pipeline to provide real-time sentiment predictions and confidence scores.

### 10. Pickle Save and Load
The trained **Bernoulli Naive Bayes** model and the **TF-IDF Vectorizer** are saved as `.pkl` files to allow for easy deployment without retraining.

### 11. Model Deployment in Flask
The project is deployed as a web application using the **Flask** framework.
* The app provides a user interface (`index.html`) where users can submit reviews.
* It processes the input, predicts sentiment, and displays the result along with the prediction confidence percentage.

**User Interface:**
![UI_1](https://github.com/avimallik/amazon_reviews_sentiment_analysis_naive_bayes/blob/master/screenshoot_app.PNG?raw=true)

![UI_2](https://github.com/avimallik/amazon_reviews_sentiment_analysis_naive_bayes/blob/master/screenshoot_app_2.PNG?raw=true)

## How to Run
1. Install dependencies: `pip install flask pandas nltk scikit-learn seaborn matplotlib`.
2. Run the training notebook or script to generate the `.pkl` files.
3. Start the Flask server: `python app.py`.
4. Open your browser to `http://127.0.0.1:5000/`.