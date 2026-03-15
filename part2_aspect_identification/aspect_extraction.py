import json
import nltk

def setup_nltk():
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger", quiet=True)


aspect_mapping = {

    "power": "performance",
    "pickup": "performance",
    "acceleration": "performance",
    "torque": "performance",
    "speed": "performance",
    "performance": "performance",
    "driving": "performance",
    "motor": "performance",

    "battery": "battery",
    "battery-life": "battery",
    "backup": "battery",

    "range": "range",
    "distance": "range",

    "charging": "charging",
    "charge": "charging",
    "charger": "charging",
    "station": "charging",
    "infrastructure": "charging",
    "network": "charging",

    "comfort": "comfort",
    "seat": "comfort",
    "seats": "comfort",
    "ride": "comfort",
    "suspension": "comfort",
    "cabin": "comfort",
    "legroom": "comfort",
    "headroom": "comfort",

    "space": "space",
    "boot": "space",
    "storage": "space",

    "safety": "safety",
    "brake": "safety",
    "braking": "safety",
    "airbag": "safety",
    "build": "safety",
    "structure": "safety",

    "design": "appearance",
    "look": "appearance",
    "style": "appearance",
    "styling": "appearance",
    "exterior": "appearance",
    "dashboard": "appearance",

    "interior": "interior",
    "material": "interior",
    "quality": "interior",
    "cabin": "interior",

    "feature": "features",
    "features": "features",
    "system": "features",
    "infotainment": "features",
    "display": "features",
    "screen": "features",
    "cluster": "features",
    "technology": "features",
    "mode": "features",
    "modes": "features",

    "price": "price",
    "cost": "price",
    "value": "price",
    "expensive": "price",
    "affordable": "price",

    "maintenance": "maintenance",
    "service": "service",
    "servicing": "service",

    "efficiency": "efficiency",
    "economy": "efficiency",
    "running": "efficiency",
    "expense": "efficiency",

    "handling": "handling",
    "steering": "handling",
    "control": "handling",
    "stability": "handling",

    "tata": "brand",
    "nexon": "brand",

    "eco": "eco-friendliness",
    "electric": "eco-friendliness",
    "ev": "eco-friendliness",
    "emission": "eco-friendliness"
}


def extract_aspects(input_path, output_path):

    with open(input_path, "r") as f:
        data = json.load(f)

    aspect_output = []

    for review in data:

        tokens = review["tokens"]

        pos_tags = nltk.pos_tag(tokens)

        aspects = []

        for word, tag in pos_tags:
            if tag.startswith("NN") and word in aspect_mapping:
                aspects.append(aspect_mapping[word])

        aspects = sorted(list(set(aspects)))

        aspect_output.append({
            "review_id": review["review_id"],
            "aspects": aspects
        })

    with open(output_path, "w") as f:
        json.dump(aspect_output, f, indent=2)

    print("Aspect extraction completed.")
    print("Saved at:", output_path)


if __name__ == "__main__":

    setup_nltk()

    input_file = r"C:\Users\KIIT0001\Aspect-Based-Sentiment-Analysis\data\preprocessed_output_withSR.json"
    output_file = r"C:\Users\KIIT0001\Aspect-Based-Sentiment-Analysis\data\aspect_output.json"

    extract_aspects(input_file, output_file)