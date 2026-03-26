import json
import nltk


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
    "time": "charging",
    "hours": "charging",

    "comfort": "comfort",
    "seat": "comfort",
    "seats": "comfort",
    "ride": "comfort",
    "suspension": "comfort",
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


def extract_aspects():

    with open("data/preprocessed_output_withSR.json", "r") as f:
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
            "vehicle_model": review["vehicle_model"],
            "aspects": aspects
        })

    with open("data/aspect_output.json", "w") as f:
        json.dump(aspect_output, f, indent=2)

    print("Aspect extraction completed.")
