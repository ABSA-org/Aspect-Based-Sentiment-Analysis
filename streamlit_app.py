import streamlit as st
import json
import pandas as pd
import plotly.express as px
import math

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="EV Sentiment Dashboard",
    layout="wide"
)

st.title("EV Aspect-Based Sentiment Analysis Dashboard")

# ---------------- LOAD REVIEWS ----------------
with open("data/raw_reviews.json", "r") as f:
    reviews = json.load(f)

vehicle_models = sorted(list(set(r["vehicle_model"] for r in reviews)))

selected_model = st.selectbox(
    "Select EV Model",
    vehicle_models
)

filtered_reviews = [
    r for r in reviews
    if r["vehicle_model"] == selected_model
]

total_reviews = len(filtered_reviews)

st.subheader(f"Vehicle: {selected_model}")
st.write(f"Total Reviews Analysed: {total_reviews}")

st.divider()

# ---------------- LOAD ASPECT SUMMARY ----------------
with open("outputs/aspect_summary.json", "r") as f:
    aspect_summary = json.load(f)

aspect_summary = aspect_summary[selected_model]

st.subheader("📊 Key Insights")

# ---------------- IMPROVED FEATURE SELECTION ----------------
most_praised_aspect = None
best_positive_score = -1

most_criticised_aspect = None
best_negative_score = -1

total_positive = 0
total_negative = 0
total_neutral = 0
total_mentions = 0

MIN_MENTIONS_THRESHOLD = 5   # ignore very rare aspects

for aspect, info in aspect_summary.items():

    counts = info["counts"]

    pos = counts["positive"]
    neg = counts["negative"]
    neu = counts["neutral"]

    total = pos + neg + neu

    if total < MIN_MENTIONS_THRESHOLD:
        continue

    total_positive += pos
    total_negative += neg
    total_neutral += neu
    total_mentions += total

    pos_pct = info["percentage"]["positive"]
    neg_pct = info["percentage"]["negative"]

    # ⭐ Weighted scoring
    weight = math.log(total + 1)

    positive_score = pos_pct * weight
    negative_score = neg_pct * weight

    if positive_score > best_positive_score:
        best_positive_score = positive_score
        most_praised_aspect = aspect
        praised_pct = pos_pct

    if negative_score > best_negative_score:
        best_negative_score = negative_score
        most_criticised_aspect = aspect
        criticised_pct = neg_pct

# fallback
if most_praised_aspect is None:
    most_praised_aspect = "N/A"
    praised_pct = 0

if most_criticised_aspect is None:
    most_criticised_aspect = "N/A"
    criticised_pct = 0

# ---------------- SCORES ----------------
overall_score = (total_positive / total_mentions) * 100

if (total_positive + total_negative) > 0:
    opinion_score = (
        total_positive /
        (total_positive + total_negative)
    ) * 100
else:
    opinion_score = 0

acceptance_score = (
    (total_positive + total_neutral) /
    total_mentions
) * 100

# ---------------- METRICS ----------------
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "⭐ Most Praised Feature",
    most_praised_aspect,
    f"{round(praised_pct,1)}% positive"
)

col2.metric(
    "⚠️ Most Criticised Feature",
    most_criticised_aspect,
    f"{round(criticised_pct,1)}% negative"
)

col3.metric(
    "😊 Overall Satisfaction",
    f"{round(overall_score,1)}%",
    help="Percentage of positive sentiment among all aspect mentions"
)

col4.metric(
    "🔥 Opinion Satisfaction",
    f"{round(opinion_score,1)}%",
    help="Positive vs negative sentiment ignoring neutral opinions"
)

col5.metric(
    "👍 Acceptance Score",
    f"{round(acceptance_score,1)}%",
    help="Percentage of non-negative user experience"
)

st.divider()

# ---------------- BAR CHART ----------------
chart_data = []

for aspect, info in aspect_summary.items():

    counts = info["counts"]
    total = counts["positive"] + counts["negative"] + counts["neutral"]

    chart_data.append({
        "aspect": aspect,
        "positive": info["percentage"]["positive"],
        "negative": info["percentage"]["negative"],
        "neutral": info["percentage"]["neutral"],
        "total_mentions": total
    })

df_chart = pd.DataFrame(chart_data)

num_aspects = st.slider(
    "Select number of aspects to display",
    min_value=3,
    max_value=len(df_chart),
    value=8
)

df_chart = df_chart.sort_values(
    by="total_mentions",
    ascending=False
).head(num_aspects)

df_melt = df_chart.melt(
    id_vars="aspect",
    value_vars=["positive", "negative", "neutral"],
    var_name="sentiment",
    value_name="percentage"
)

fig = px.bar(
    df_melt,
    x="aspect",
    y="percentage",
    color="sentiment",
    barmode="group",
    title="Aspect-wise Sentiment Distribution (%)",
    height=500
)

st.plotly_chart(fig, width="stretch")

st.divider()

# ---------------- TABLE ----------------
st.subheader("Aspect Sentiment Detailed Table")

table_data = []

for aspect, info in aspect_summary.items():

    counts = info["counts"]
    perc = info["percentage"]

    table_data.append({
        "Aspect": aspect,
        "Positive Count": counts["positive"],
        "Negative Count": counts["negative"],
        "Neutral Count": counts["neutral"],
        "Positive %": perc["positive"],
        "Negative %": perc["negative"],
        "Neutral %": perc["neutral"],
        "Final Sentiment": info["final_sentiment"]
    })

df_table = pd.DataFrame(table_data)

df_table = df_table.sort_values(
    by="Positive %",
    ascending=False
).reset_index(drop=True)

st.dataframe(
    df_table,
    width="stretch",
    height=400
)

st.divider()

# ---------------- DRILL DOWN ----------------
st.subheader("🔎 Aspect Drill-Down Viewer")

with open("outputs/final_aspect_sentiment.json", "r") as f:
    review_level_data = json.load(f)

with open("data/raw_reviews.json", "r") as f:
    raw_reviews = json.load(f)

review_lookup = {
    r["review_id"]: r["review_text"]
    for r in raw_reviews
}

aspect_list = sorted(aspect_summary.keys())

selected_aspect = st.selectbox(
    "Select Aspect",
    aspect_list
)

sentiment_option = st.radio(
    "Filter by Sentiment",
    ["All", "Positive", "Negative", "Neutral"],
    horizontal=True
)

filtered = []

for item in review_level_data:

    if item["vehicle_model"] != selected_model:
        continue

    aspect_dict = item["aspect_sentiment"]

    if selected_aspect in aspect_dict:

        sentiment = aspect_dict[selected_aspect]

        if sentiment_option == "All" or sentiment.lower() == sentiment_option.lower():
            filtered.append({
                "review_id": item["review_id"],
                "sentiment": sentiment
            })

total_reviews = len(filtered)

st.write(f"Total matching reviews: {total_reviews}")

if total_reviews == 0:
    st.warning("No reviews found for selected filters")

else:

    if total_reviews <= 5:
        num_to_show = total_reviews
    else:
        num_to_show = st.slider(
            "Number of reviews to display",
            min_value=5,
            max_value=min(total_reviews, 100),
            value=min(10, total_reviews)
        )

    filtered_to_show = filtered[:num_to_show]

    for r in filtered_to_show:

        review_id = r["review_id"]
        sentiment = r["sentiment"]

        review_text = review_lookup.get(
            review_id,
            "Review text not found"
        )

        display_text = f"Review {review_id}: {review_text}"

        if sentiment == "positive":
            st.success(display_text)
        elif sentiment == "negative":
            st.error(display_text)
        else:
            st.info(display_text)