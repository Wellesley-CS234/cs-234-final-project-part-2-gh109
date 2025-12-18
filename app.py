import streamlit as st
import pandas as pd
import altair as alt
import zipfile
import os


zip_path = "physics_pageviews.zip"
extract_to = "."   

# Unzip if not already extracted
if not os.path.exists("physics_pageviews.csv"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Unzipped physics_pageviews.csv")
else:
    print("File already unzipped.")

st.set_page_config(page_title="Physics Wikipedia Analytics", layout="wide")

@st.cache_data


def load_data():
    daily = pd.read_csv("daily_views.csv")
    article_totals = pd.read_csv("article_views.csv")
    full = pd.read_csv("physics_pageviews.csv")

    # ensure dates are parsed correctly
    daily["date"] = pd.to_datetime(daily["date"])
    full["date"] = pd.to_datetime(full["date"])

    return daily, article_totals, full


# Load data
daily_views, article_totals, full_data = load_data()

st.title("Physics Wikipedia Analytics")
st.subheader('RQ: Are "advance physics concepts" visited more than "classical physics concepts" adter Nobel Prize Annoucements?')

# =======================================================
# 1. DAILY TIME SERIES (AGGREGATE)
# =======================================================

st.header("Daily Total Pageviews")

daily_chart = (
    alt.Chart(daily_views)
    .mark_line()
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("pageviews:Q", title="Total Pageviews", scale=alt.Scale(zero=True)),
        tooltip=["date:T", "pageviews:Q"]
    )
    .properties(height=400)
    .interactive()
)

st.altair_chart(daily_chart, use_container_width=True)


# =======================================================
# 2. TOP ARTICLES BAR CHART
# =======================================================

st.header("Top Physics Articles by Total Pageviews")

top_n = st.slider("Number of articles to display:", 5, 50, 10)

top_articles = article_totals.sort_values("pageviews", ascending=False).head(top_n)

bar_chart = (
    alt.Chart(top_articles)
    .mark_bar()
    .encode(
        x=alt.X("pageviews:Q", title="Total Pageviews"),
        y=alt.Y("article:N", sort="-x", title="Article"),
        tooltip=["article", "pageviews"]
    )
    .properties(height=400)
)

st.altair_chart(bar_chart, use_container_width=True)


# =======================================================
# 3. INDIVIDUAL ARTICLE VIEWER
# =======================================================

st.header("Explore an Article's Pageview History")

article_list = sorted(full_data["article"].unique())
# Sidebar search box
search = st.text_input("Search for an article:", "")

# Filter articles based on search term
if search:
    filtered_articles = [a for a in article_list if search.lower() in a.lower()]
else:
    filtered_articles = article_list

# Selectbox with filtered results
chosen_article = st.selectbox("Choose an article:", filtered_articles)

# Plot
article_df = full_data[full_data["article"] == chosen_article]

article_chart = (
    alt.Chart(article_df)
    .mark_line(color="#0066cc")
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("pageviews:Q", title="Pageviews"),
        tooltip=["date:T", "pageviews:Q"]
    )
    .properties(height=300)
    .interactive()
)

st.altair_chart(article_chart, use_container_width=True)

