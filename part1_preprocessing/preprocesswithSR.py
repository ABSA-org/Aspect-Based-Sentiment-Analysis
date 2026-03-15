import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def setup_nltk():
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet"
    }

    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess(text):

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def preprocess_reviews():

    input_path = r"C:\Users\KIIT0001\Aspect-Based-Sentiment-Analysis\data\raw_reviews.json"
    output_path = r"C:\Users\KIIT0001\Aspect-Based-Sentiment-Analysis\data\preprocessed_output_withSR.json"

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

    print("Preprocessing completed.")
    print("Saved at:", output_path)

setup_nltk()
preprocess_reviews()