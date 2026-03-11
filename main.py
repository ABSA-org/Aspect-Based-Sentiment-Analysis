import json
from collections import defaultdict
from part3_sentiment_analysis.sentiment_analysis import run_sentiment_analysis

#prep,aspect_identification,sentiment_analysis





#sentiment_analysis
run_sentiment_analysis()

#aggregation
with open("outputs/final_aspect_sentiment.json","r") as f:
    data = json.load(f)

aspect_counts = defaultdict(lambda: {"positive":0,"negative":0,"neutral":0})

for review in data:
    aspects = review["aspect_sentiment"]

    for aspect, sentiment in aspects.items():
        aspect_counts[aspect][sentiment] += 1

aspect_summary = {}

for aspect, counts in aspect_counts.items():
    total = sum(counts.values())
    pos_pct = (counts["positive"] / total) * 100
    neg_pct = (counts["negative"] / total) * 100
    neu_pct = (counts["neutral"] / total) * 100

    final_sentiment = max(counts, key=counts.get)

    aspect_summary[aspect] = {
        "counts": counts,
        "percentage": {
            "positive": round(pos_pct,2),
            "negative": round(neg_pct,2),
            "neutral": round(neu_pct,2)
        },
        "final_sentiment": final_sentiment
    }

with open("outputs/aspect_summary.json","w") as f:
    json.dump(aspect_summary, f, indent=4)

print("Aspect Sentiment Summary Generated")