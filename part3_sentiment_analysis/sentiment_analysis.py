import json
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
def dependency_parser_sentiment(doc, aspect, sia, intensifiers):
    parser_score = 0
    for token in doc:

        if token.text.lower() == aspect.lower():

            for child in token.children:

                if child.pos_ in ["ADJ", "VERB", "ADV"]:

                    score = sia.polarity_scores(child.text)["compound"]

                    for modifier in child.children:
                        if modifier.text.lower() in intensifiers:
                            score *= intensifiers[modifier.text.lower()]

                    for neg in child.children:
                        if neg.dep_ == "neg":
                            score = -score

                    parser_score += score

            if token.head.pos_ in ["ADJ", "VERB", "ADV"]:

                parent_word = token.head.text
                score = sia.polarity_scores(parent_word)["compound"]

                for modifier in token.head.children:
                    if modifier.text.lower() in intensifiers:
                        score *= intensifiers[modifier.text.lower()]

                for neg in token.head.children:
                    if neg.dep_ == "neg":
                        score = -score

                parser_score += score

    return parser_score


def context_window_sentiment(tokens, aspect, sia, window_size=3):

    window_score = 0

    for i, word in enumerate(tokens):

        if word.lower() == aspect.lower():

            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)

            window = tokens[start:end]

            for w in window:
                score = sia.polarity_scores(w)["compound"]
                window_score += score

    return window_score





def run_sentiment_analysis():

    nltk.download('vader_lexicon')
    nlp = spacy.load("en_core_web_sm")
    sia = SentimentIntensityAnalyzer()

    INTENSIFIERS = {
        "very": 1.5,
        "extremely": 2.0,
        "really": 1.5,
        "too": 1.3,
        "quite": 1.2,
        "highly": 1.7,
        "slightly": 0.5,
        "somewhat": 0.7
    }

    with open("test_data/test_preprocessed_output_withoutSR.json", "r") as f:
        reviews = json.load(f)

    with open("test_data/test_aspect_output.json", "r") as f:
        aspects_data = json.load(f)

    final_results = []

    for review_obj in reviews:

        review_id = review_obj["review_id"]
        tokens = review_obj["preprocessed_review"]

        aspect_list = []
        for asp in aspects_data:
            if asp["review_id"] == review_id:
                aspect_list = asp["aspects"]
                break

        sentence = " ".join(tokens)
        doc = nlp(sentence)

        aspect_sentiments = {}

        for aspect in aspect_list:

            parser_score = dependency_parser_sentiment(doc, aspect, sia, INTENSIFIERS)
            window_score = context_window_sentiment(tokens, aspect, sia)

            # Hybrid score
            final_score = parser_score + 0.5 * window_score

            if final_score > 0:
                sentiment = "positive"
            elif final_score < 0:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            aspect_sentiments[aspect] = sentiment

        final_results.append({
            "review_id": review_id,
            "aspect_sentiment": aspect_sentiments
        })

    with open("outputs/final_aspect_sentiment.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("Hybrid Aspect Sentiment Analysis Completed.")