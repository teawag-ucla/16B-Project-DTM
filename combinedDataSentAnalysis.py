from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
sia = SentimentIntensityAnalyzer()

all_data_df_edited = all_data_df.copy()
all_data_df_edited["sentiment_score"] = all_data_df_edited["Headline"].apply(lambda x: sia.polarity_scores(x)["compound"])

#clean the next to then help see frequencies in words
import re

#makeing the headline and byline into one
all_data_df_edited["combined_text"] = all_data_df_edited["Headline"].fillna('') + " " + all_data_df_edited["Byline"].fillna('')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation and numbers
    return text


all_data_df_edited["cleaned_text"] = all_data_df_edited["combined_text"].apply(clean_text)

#see the frequencies in the words - FOR ALL OF OUR DATA COMBINED
from collections import Counter
from nltk.corpus import stopwords

nltk.download('stopwords')
english_stopwords = set(stopwords.words('english'))

all_words = " ".join(all_data_df_edited["cleaned_text"]).split()
filtered_words_nonstopwords = [word for word in all_words if word not in english_stopwords]
word_freq = Counter(filtered_words_nonstopwords)
common_words = word_freq.most_common(50)
#print(common_words)

import seaborn as sns
import matplotlib.pyplot as plt
word_freq_df = pd.DataFrame(common_words, columns=["word", "count"])
plt.figure(figsize=(10, 12))
sns.barplot(x="count", y="word", data=word_freq_df.head(30))

#which pairs of words are popular
from sklearn.feature_extraction.text import CountVectorizer #converts to numerical counts

#phrases (two word ones)
phrases_two_words = CountVectorizer(ngram_range=(2,2), stop_words="english", max_features=60)
#learns the vocabulary of bigrams from cleaned_text col and transforms each document into a numerical vector
X2 = phrases_two_words.fit_transform(all_data_df_edited["cleaned_text"])
phrases = pd.DataFrame({
    "phrases": phrases_two_words.get_feature_names_out(),
    "count": X2.toarray().sum(axis=0) #sums the counts: gives total frequency of each bigram
}).sort_values("count", ascending=False)
phrases.head(20)
plt.figure(figsize=(10, 12))
sns.barplot(x="count", y="phrases", data=phrases.head(20))

#see the frequencies in the words - FOR JUST NEWS
df_combined["combined_text"] = df_combined["Headline"].fillna('') + " " + df_combined["Byline"].fillna('')
df_combined["cleaned_text"] = df_combined["combined_text"].apply(clean_text).apply(split_merged_words)

all_words = " ".join(df_combined["cleaned_text"]).split()
filtered_words_nonstopwords = [word for word in all_words if word not in english_stopwords]
word_freq = Counter(filtered_words_nonstopwords)
common_words = word_freq.most_common(50)
#print(common_words)

import seaborn as sns
import matplotlib.pyplot as plt
word_freq_df = pd.DataFrame(common_words, columns=["word", "count"])
plt.figure(figsize=(10, 12))
sns.barplot(x="count", y="word", data=word_freq_df.head(30))

#which pairs of words are popular
from sklearn.feature_extraction.text import CountVectorizer #converts to numerical counts

#phrases (two word ones)
phrases_two_words = CountVectorizer(ngram_range=(2,2), stop_words="english", max_features=60)
#learns the vocabulary of bigrams from cleaned_text col and transforms each document into a numerical vector
X2 = phrases_two_words.fit_transform(df_combined["cleaned_text"])
phrases = pd.DataFrame({
    "phrases": phrases_two_words.get_feature_names_out(),
    "count": X2.toarray().sum(axis=0) #sums the counts: gives total frequency of each bigram
}).sort_values("count", ascending=False)
phrases.head(20)
plt.figure(figsize=(10, 12))
sns.barplot(x="count", y="phrases", data=phrases.head(20))

#update the sentiment scores based on sustainable contexts/terminologies
new_words = {
    "sustainability": 2.0,
    "sustainable": 2.0,
    "greenwashing": -2.5,
    "carbon-neutral": 1.5,
    "renewable": 1.8,
    "eco-friendly": 1.8,
    "pollution": -2.0,
    "emissions": -1.5,
    "recycling": 1.3,
    "green": 1.2,
    "cruelty-free": 1.5,
    "zero-waste": 1.5,
    "carbon": 1,
    "energy": 1,
    "climate": 1.5,
    "solar": 1.6,
    "power": 1.1,
    "water": 1.1,
    "wind": 1.3,
    "waste": -1.2,
    "gas": -1.3,
    "electricity": 1.2,
    "fossil": -2,
    "clean": 2,
    "renewable energy": 2,
    "climate change": 1.7,
    "fossil fuels": -1.7,
    "green energy": 2,
    "solar farm": 1.7,
    "greenhouse gas": -1.6,
    "clean energy": 1.4,
    "sustainable consumer": 1.5 ,
    "sustainable developement": 1.6,
    "sustainable consumption": 1.4
}

sia.lexicon.update(new_words)
#now scoring this with the updated values for having sustainable contexts
all_data_df_edited["sustainable_updated_sentiment"] = all_data_df_edited["combined_text"].apply(lambda x: sia.polarity_scores(x)["compound"])

#now cluster it using KMeans to see the trends
#this can tell us what types of headlines are most commonly used
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(all_data_df_edited["Headline"])

kmeans = KMeans(n_clusters=20, random_state=42)
all_data_df_edited["cluster"] = kmeans.fit_predict(X)

#looking into the common words per each cluster
#the words that the tfidfVectrozier has already pulled from the headlines/bylines
terms_to_look_through = np.array(tfidf.get_feature_names_out())

#what words are most commonly used in each cluster - can give an overview of what kinds of headlines are in each cluster
for i in range(kmeans.n_clusters):
    center = kmeans.cluster_centers_[i]
    indicies_with_common_terms = center.argsort()[::-1] #returns indicies from high to low
    common_terms_per_cluster = terms_to_look_through[indicies_with_common_terms][:15]



#per cluster what is the average sentiment (whihc comes from the updated sentiment scores)
cluster_summary = (all_data_df_edited.groupby("cluster").agg(avg_sentiment=("sustainable_updated_sentiment", "mean"),count=("Headline", "count"))
.reset_index())

#where its like df in a df
def get_top_headlines(df, cluster_num, sentiment_col="sustainable_updated_sentiment", n=5): #can change n parameter
    look_into_each_cluster = df[df["cluster"] == cluster_num]
    #nlargest: returns the first n rows of a DataFrame with the largest values in specified columns, ordered in descending order.
    top_pos = look_into_each_cluster.nlargest(n, sentiment_col)[["Headline", sentiment_col]]
    #top_neg = sub.nsmallest(n, sentiment_col)[["Headline", sentiment_col]]
    return top_pos#, top_neg

cluster_summary["top_positive"] = cluster_summary["cluster"].apply(
    lambda cl: get_top_headlines(all_data_df_edited, cl)
)


#filtering the top cluster terms by sustainability related terms
words_to_filter_by = [
    "sustainable", "renewable", "eco", "green", "carbon",
    "emission", "energy", "recycle", "marketing", "innovation", "environment", "climate", "change", "emissions", "greenhouse", "farming", "solar", "farm",
    "recycled", "water", "light", "greener", "recycling", "waste", "food", "plastic", "gas", "natural", "clean", "power"]

filtered_clusters = {}

for i in range(kmeans.n_clusters):
    center = kmeans.cluster_centers_[i]
    indicies_with_common_terms = center.argsort()[::-1] #returns indicies from high to low
    common_terms_per_cluster = terms_to_look_through[indicies_with_common_terms][:15]
    #now filter these cluster terms to the "sustainable" context
    filtered_terms = [term for term in common_terms_per_cluster
                      if any(f in term for f in words_to_filter_by)]
    filtered_clusters[i] = filtered_terms #make that index (so cluster in this case) have those filtered terms

  summary = (all_data_df_edited.groupby("cluster").agg(avg_sentiment=("sustainable_updated_sentiment", "mean"),count=("Headline", "count")).reset_index())

#replaces each cluster number with the list of filtered sustainability terms per cluster
summary["top_focus_terms_green_related"] = summary["cluster"].map(filtered_clusters)
summary.loc[1, "top_focus_terms_green_related"] = ["sustainable"] #to fix row
summary.loc[4, "top_focus_terms_green_related"] = ["recycling"] #to fix row
#count how many sustainable related terms are in each row (by checking if it is a list, then finding the length of it)
summary["focus_term_count"] = summary["top_focus_terms_green_related"].apply(lambda x: len(x) if isinstance(x, list) else 0)

#visuals for above
#see how frequent those filter terms are
#explode: transform each element of a list-like to a row, replicating index values.
  #makes it easier to go through and really see each one better (to spilt it up)
summary_exploded = summary.explode("top_focus_terms_green_related")
summary_exploded = summary_exploded.dropna()

focus_term_summary = (
    summary_exploded.groupby("top_focus_terms_green_related")
    .agg(
        frequency=("cluster", "count"),
        avg_sentiment=("avg_sentiment", "mean")
    )
    .sort_values("frequency", ascending=False)
)

#visualizing these terms and their avg sentiment
top_terms = focus_term_summary
sns.barplot(y=top_terms.index, x=top_terms["avg_sentiment"], hue = top_terms.index, legend=False)
plt.title("Average Sentiment by Common Sustainability Term")
plt.xlabel("Average Sentiment")
plt.ylabel("Sustainability Keyword")
plt.show()

#visualizing relationshop between amount of sustainability related terms and the sentiment
import matplotlib.pyplot as plt

plt.scatter(summary["focus_term_count"], summary["avg_sentiment"], s=summary["count"]*5, alpha=0.7)
plt.xlabel("Number of Sustainability Terms (Focus Term Count)")
plt.ylabel("Average Sentiment")
plt.title("Relationship Between Focus Term Frequency and Sentiment")
plt.show() 

#exploring sentiment by categories
#explorining sentiment by keywords
sentiment_overview_by_keyword = (
    all_data_df_edited.groupby("Keyword")["sustainable_updated_sentiment"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
)

#explorining sentiment by source type
sentiment_overview_source = (
    all_data_df_edited.groupby("Source Type")["sustainable_updated_sentiment"]
    .agg(["mean", "median", "count"])
)

#most Positive vs most Negative Words (word + sentiment correlation)
tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(all_data_df_edited["cleaned_text"])
terms = tfidf.get_feature_names_out() #the unique words that tfidf gets out from the cleaned_text column

from scipy.stats import pearsonr
word_corr = []

for idx, term in enumerate(terms):
    word_weight = X[:, idx].toarray().flatten() #give tfidf score of this specific word we are on across all headlines/cleaned text
    #pearsonr: measures the linear relationship between two datasets
    corr, _ = pearsonr(word_weight, all_data_df_edited["sustainable_updated_sentiment"]) #pearson R: can get the correlation b/t word_weight and scores of the headlines the specific word is in
    word_corr.append((term, corr)) #now the word_corr can be more thorough to show for analysis

positive_words = sorted(word_corr, key=lambda x: x[1], reverse=True)[:20]  # positive words
negative_words = sorted(word_corr, key=lambda x: x[1])[:20]                # negative words

#seeing them visually
#convert both to dataframe
df_pos = pd.DataFrame(positive_words, columns=["word", "correlation"])
df_neg = pd.DataFrame(negative_words, columns=["word", "correlation"])

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

#opsitive words
axes[0].barh(df_pos["word"][::-1], df_pos["correlation"][::-1], color="green")
axes[0].set_title("Top 20 Words Positively Correlated\nwith Sustainability Sentiment")
axes[0].set_xlabel("Correlation")
axes[0].set_ylabel("Word")

# negative words
axes[1].barh(df_neg["word"][::-1], df_neg["correlation"][::-1], color="red")
axes[1].set_title("Top 20 Words Negatively Correlated\nwith Sustainability Sentiment")
axes[1].set_xlabel("Correlation")
axes[1].set_ylabel("Word")

plt.show()