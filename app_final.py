import streamlit as st
import pandas as pd
import altair as alt
import re

st.set_page_config(
    page_title="Physics Wikipedia Project",
    layout="wide"
)

st.title("Physics Wikipedia Articles: Subfield Classification & Analysis")
tab_intro, tab_features, tab_results, tab_viz = st.tabs([
    "📘 Introduction & Data Summary",
    "🧠 New Features & Text Classification",
    "📊 Results",
    "📈 Visualizations & Summary"
])

with tab_intro:
    
    st.header("Research Question: Do certain subfields of physics get more views than others?")

    st.markdown("""
   
    **Motivation:**  
    As a physics major, I was curious to see if certain fields of physics are more popular than others.
    I defined a list of subfields of physics based on the WikiProject for physics and some personal adjustments, then compared 
                them to see which fields were the most viewed. 

    **Hypotheses:**  
    - Certain subfields (e.g. classical mechanics, which is considered the starting point in physics for many people)
      receive high attention.
    - On the other hand, more "advanced" subfields, which people might not have heard about, recieve less attention.
                
    **Results:**
    - More well known subject do recieve relatively more views compared to more advanced subjects.
    - However a lot of broader, within subject articles got missclassified, which could be throwing off the overall proportions.
    """)



    st.header("Data Summary")

    st.markdown("""
    **Data source:**  
    - Wikipedia Physics Project articles and pageview data from **3/7/23–12/30/24**
    - Total Number of Articles: **28391**
    - Articles that got views in 2023-2024: **4762**
    - Percentage of articles that got views: **~16.77%**
    
    The classification was done only on pages the recieved views and had an *instance of* attribute.


    **Metadata used:**  
    - Article descriptions  
    - Wikidata attributes (instance of, part of, subclass of, etc.)
    """)


    # Example: load your final CSV
    df = pd.read_csv("articles_with_predictions.csv")

    st.subheader("Preview of Dataset")
    st.dataframe(df)

with tab_features:
    st.header("New Features")
    st.subheader("Using metadata from Wikidata, I added some attributes of the articles with views.")
    st.markdown(""" *I refiltered the articles with views so that only the ones with the attribute **instance of** were kept,
                 since I used them as ground-truth labels.*  """)
    
    st.markdown("""

    **Added features:**
    - instance of
    - part of
    - subclass of
    - facet of
    - has characteristic
    """)

    st.markdown("**First 10 rows with added features**")
    st.write(df.iloc[:10, [0, 1, 2, 3, 4, 5, 6, 7]])

    st.subheader("Text Classification")
    st.markdown("""
    To obtain a ground truth, I got a list of every unique *instance of* and gave it to Gemini. Then I requested it to classify each *instance of* into one of the following physics subfields:

    - Classical mechanics
    - Quantum physics
    - Thermodynamics
    - Particle physics
    - Condensed matter
    - Relativity
    - Atomic, molecular, and optical physics
    - Physicist
    - Astronomy and Planetary
    - Mathematical Concept
    - Technology
    - Other
    """)
    st.markdown("""These were determined partly by the stubs section of the physics WIkiproject, where there was a list of subclassifications.
                I added *Astronomy and Planetary, Mathematical Concept, Technology, and Other* to get the LLM to be more accurate with it classifications,
                instead of putting so many things in *other*.""" )
    
    df_classes = pd.read_csv("subclasses.csv")
    st.write("10 random rows from instance of to subclass mapping:")
    st.write(df_classes.sample(n=10))

    st.markdown("""
    Then I trained a **Naive Bayes classifier** to assign each article
    to one of the subfields by combining the text from their description. If they had additional attribute in *subclass of*, 
                *facet of, part of* and/or *has characteristic*, that was also combined with the description to help make the predictions.
    
                
    """)

    st.markdown("""
    **Training data:**  
    Ground-truth labels derived from Wikidata *instance of* values.
    Trained on randomly selected 80\% of data, and saved 20\% for testing. 

    **Evaluation:**  
    - Accuracy: **0.731**
    - F1 Score: **0.674**
                
    """)
    st.image("confusion_matrix.png")

    st.write(df.sample(n=10))


    st.subheader("Predicted Subfield Distribution")
    exclude_other = st.checkbox("Exclude 'Other' subclass")
    if exclude_other:
        data = df[df["predicted_subclass"] != "Other"]["predicted_subclass"].value_counts()
    else:
        data = df["predicted_subclass"].value_counts()


    st.bar_chart(data)

with tab_results:
    st.header("Results")

    st.markdown("""
    Since the data spans a two-year period (2023–2024), we compare subfields
    visually rather than using hypothesis testing with p-values.
    """)
    st.markdown("""Overall, we can see that mathematical concepts get the most views. However, some subfields such as relativity get relatively more views
                compared to their proportion of total articles. On the other hand, some fields like optics, get lower views compared to their proportion. """)

    
    exclude_other = st.checkbox("Exclude 'Other' subclass  ")
    exclude_physicist = st.checkbox("Exclude 'Physicist' subclass  ")

    filtered_articles = df.copy()

    if exclude_other:
        filtered_articles = filtered_articles[
            filtered_articles["ground_truth_subclass"] != "Other"
        ]

    if exclude_physicist:
        filtered_articles = filtered_articles[
            filtered_articles["ground_truth_subclass"] != "Physicist"
        ]

  
    st.subheader("Proportion of Articles by Subfield (Ground Truth)")

    article_props = (
        filtered_articles["ground_truth_subclass"]
        .value_counts(normalize=True)
    )

    st.bar_chart(article_props)
    
    pageviews_df = pd.read_csv(
        "physics_pageviews.csv",
        parse_dates=["date"]
    )

    pageviews_df["qid"] = pageviews_df["qid"].astype(str)
    df["QID"] = df["QID"].astype(str)

    
    merged_views = pageviews_df.merge(
        df[["QID", "label", "ground_truth_subclass"]],
        left_on="qid",
        right_on="QID",
        how="left"
    )

    
    if exclude_other:
        merged_views = merged_views[
            merged_views["ground_truth_subclass"] != "Other"
        ]

    if exclude_physicist:
        merged_views = merged_views[
            merged_views["ground_truth_subclass"] != "Physicist"
        ]

    
    st.subheader("Total Pageviews by Subfield (2023–2024)")

    views_by_subfield = (
        merged_views
        .groupby("ground_truth_subclass")["pageviews"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(views_by_subfield)

    views_by_subfield_prop = views_by_subfield / views_by_subfield.sum()
    st.subheader("Proportion of Total Pageviews by Subfield (2023–2024)")
    st.markdown(""" This shows the *share of total physics-related pageviews* attributed to each subfield over the 2023–2024 period.""")


    st.bar_chart(views_by_subfield_prop)
    


    #st.dataframe(unmatched_articles, use_container_width=True)



with tab_viz:
    st.header("Interactive Exploration")
    
    selected_subfield = st.selectbox(
        "Select a physics subfield:",
        (df["ground_truth_subclass"].unique())
    )

    filtered = df[df["ground_truth_subclass"] == selected_subfield]

    st.write(f"Articles in **{selected_subfield}**:")
    st.dataframe(filtered[["label", "description"]])

    st.header("Daily Total Pageviews 2023-2024")

    daily_views = pd.read_csv("daily_views.csv")

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

    pageviews_df = pd.read_csv("physics_pageviews.csv", parse_dates=["date"])
    articles_df = pd.read_csv("articles_with_predictions.csv")

    pageviews_df["qid"] = pageviews_df["qid"].astype(str)
    articles_df["QID"] = articles_df["QID"].astype(str)
    merged_views = pageviews_df.merge(
    articles_df[["QID", "ground_truth_subclass"]],
    left_on="qid",
    right_on="QID",
    how="left"
    )

    daily_subclass_views = (
    merged_views
    .groupby(["date", "ground_truth_subclass"], as_index=False)
    .agg({"pageviews": "sum"})
    )

    st.subheader("Daily Pageviews by Physics Subfield")

    subclasses = sorted(daily_subclass_views["ground_truth_subclass"].unique())
    selected_subclass = st.selectbox(
        "Choose a subfield:",
        subclasses
    )

    plot_df = daily_subclass_views[
        daily_subclass_views["ground_truth_subclass"] == selected_subclass
    ]

    subclass_chart = (
    alt.Chart(plot_df)
    .mark_line()
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("pageviews:Q", title="Total Pageviews", scale=alt.Scale(zero=True)),
        tooltip=["date:T", "pageviews:Q"]
    )
    .properties(height=400)
    .interactive()
    )

    st.altair_chart(subclass_chart, use_container_width=True)

    article_totals = (
    merged_views
    .groupby(
        ["qid", "article", "ground_truth_subclass"],
        as_index=False)
    .agg({"pageviews": "sum"}))

    top_articles_df = (
    article_totals[
        article_totals["ground_truth_subclass"] == selected_subclass
    ]
    .sort_values("pageviews", ascending=False).head(10))

    top_articles_df["article"] = (
    top_articles_df["article"]
    .str.replace("_", " ", regex=False))

    st.subheader(f"Top 10 Articles in {selected_subclass}")

    st.dataframe(
        top_articles_df[["article", "pageviews"]],
        use_container_width=True)









    st.header("Summary, Limitations")

    st.markdown("""
    **Key takeaways:**
    - Our results seem to suggest that yes, certain subclasses of physics do get more views than others! For more niche subfields, there seems to be
                more relative views for ones that can be considered more "common", and less relative views on those considered less common."
    
    **Limitations:**
    - These subfield classifications were created with subjectivity, of both Gemini and myself. While I started with the subfields listed on the Wikiproject,
                I added more subfields I deemed relavent. Articles can be reclassified/assigned to different subfields, especially if they are considered more interdisiplinary
    - Since the ground truth was created from the *instance of* attribute, quite a few articles got classified as **other** instead of what they actually should be. 
                For example, *Quantum Mechanics* got classified into *other* since it was and instance of "physical theory" and "branch of physics". This was probably
                too broad of a term for the AI to classify. So a lot of these larger field terms ended up misclassified. 
    - Pageviews, while they can be indicitive of interest in the actual subject of physics, can still be correlated to other things. The most glaring example is the page for
                *Robert Oppenheimer*, where the summer of the movie release triggered a huge spike in pageviews for him and other related physicists. 
    

    """)
