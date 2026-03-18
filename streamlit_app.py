import streamlit as st
import json
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="EV Sentiment Dashboard",
    layout="wide"
)

st.title("EV Aspect-Based Sentiment Analysis Dashboard")

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

with open("outputs/aspect_summary.json", "r") as f:
    aspect_summary = json.load(f)

st.subheader("📊 Key Insights")

most_praised_aspect = None
max_positive = -1

most_criticised_aspect = None
max_negative = -1

total_positive = 0
total_mentions = 0

for aspect, info in aspect_summary.items():

    pos_pct = info["percentage"]["positive"]
    neg_pct = info["percentage"]["negative"]

    counts = info["counts"]
    total = counts["positive"] + counts["negative"] + counts["neutral"]

    total_positive += counts["positive"]
    total_mentions += total

    if pos_pct > max_positive:
        max_positive = pos_pct
        most_praised_aspect = aspect

    if neg_pct > max_negative:
        max_negative = neg_pct
        most_criticised_aspect = aspect

overall_score = (total_positive / total_mentions) * 100

col1, col2, col3 = st.columns(3)

col1.metric(
    "⭐ Most Praised Feature",
    most_praised_aspect,
    f"{round(max_positive,1)}% positive"
)

col2.metric(
    "⚠️ Most Criticised Feature",
    most_criticised_aspect,
    f"{round(max_negative,1)}% negative"
)

col3.metric(
    "😊 Overall Satisfaction",
    f"{round(overall_score,1)}%"
)

st.divider()

with open("outputs/aspect_summary.json", "r") as f:
    aspect_summary = json.load(f)

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
        st.write(f"Showing all {total_reviews} reviews")
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