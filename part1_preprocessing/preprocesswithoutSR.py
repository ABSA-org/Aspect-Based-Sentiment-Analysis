import json
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocess(text):

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)

    tokens = word_tokenize(text)

    tokens = [t for t in tokens if t.strip() != ""]

    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def preprocess_reviews_withoutSR():

    input_path = "data/raw_reviews.json"
    output_path = "data/preprocessed_output_withoutSR.json"

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

    print("Preprocessing without SR completed.")
