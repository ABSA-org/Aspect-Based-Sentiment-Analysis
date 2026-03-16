import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = None
lemmatizer = None

def setup_nltk():
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng"
    }

    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)

    global stop_words, lemmatizer

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()



def preprocess(text):

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def preprocess_reviews_withSR():

    input_path = "data/raw_reviews.json"
    output_path = "data/preprocessed_output_withSR.json"

    with open(input_path, "r") as f:
        reviews = json.load(f)

    preprocessed_data = []

    for review in reviews:

        processed_tokens = preprocess(review["review_text"])

        preprocessed_data.append({
            "review_id": review["review_id"],
            "vehicle_model": review["vehicle_model"],
            "tokens": processed_tokens
        })

    with open(output_path, "w") as f:
        json.dump(preprocessed_data, f, indent=2)

    print("Preprocessing with SR completed.")
