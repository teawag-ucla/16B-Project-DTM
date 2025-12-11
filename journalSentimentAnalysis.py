"""
Sources: 
* https://www.geeksforgeeks.org/python/regular-expression-python-examples/ 
* https://pymupdf.readthedocs.io/en/latest/tutorial.html
* https://pypi.org/project/pdfplumber/ 
* https://youtube.com/playlist?list=PLHlrXTRZkTLBQ7k06CFoP3-gayAmfp-GG&si=IfeGKA53SXSRNzj9
* https://pymupdf.readthedocs.io/en/latest/about.html
* https://www.nltk.org/
* https://www.geeksforgeeks.org/python/python-sentiment-analysis-using-vader/
* https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html
* https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html
* https://docs.python.org/3/library/unittest.html
"""

#Keywords list 
keywords = [
    "sustainability",
    "sustainable",
    "greenwashing",
    "carbon-neutral",
    "renewable",
    "eco-friendly",
    "pollution",
    "emissions",
    "recycling",
    "green",
    "cruelty-free",
    "zero-waste",
    "carbon",
    "energy",
    "climate",
    "solar",
    "power",
    "water",
    "wind",
    "waste",
    "gas",
    "electricity",
    "fossil",
    "clean",
    "renewable energy",
    "climate change",
    "fossil fuels",
    "green energy",
    "solar farm",
    "greenhouse gas",
    "clean energy",
    "sustainable consumer",
    "sustainable developement",
    "sustainable consumption"
]

def text_sentimentAnalysis(df: pd.DataFrame, column: str = "byline") -> pd.DataFrame:
    """
    Performs sentiment analysis on text. 

    Parameters:
    df(pd.dataframe): Dataframe containing the text (use functions pdf_scraper and journalpdf_concat from webscraping py
                                                    to get dataframe)
    column: Name of the column containing text to analyze

    Returns:
    pd.DataFrame: Original DataFrame with another column that sotres the sentiment scores
    """
    #Initialize the analyzer 
    sia = SentimentIntensityAnalyzer()

    #Adjust the dataframe 
    df = df.copy()
    df['sent_scores'] = df[column].apply(lambda text: sia.polarity_scores(str(text)))

    #Get the sentiment scores 
    df['sent_comp'] = df['sent_scores'].apply(lambda x: x['compound'])
    df['sent_pos'] = df['sent_scores'].apply(lambda x: x['pos'])
    df['sent_neg'] = df['sent_scores'].apply(lambda x: x['neg'])
    df['sent_neu'] = df['sent_scores'].apply(lambda x: x['neu'])

    #Column with the type of sentiment labelled to the text
    df['sent_type'] = df['sent_comp'].apply(lambda x: 'pos' if x >= 0.05 else 'neg' if x <= -0.05 else 'neu')

    return df

def key_sentimentAnalysis(df: pd.DataFrame, column: str = "byline", keyCol = "keyword") -> pd.DataFrame:
    """
    Analyzes sentiment patterns for different keywords that were present in the journal. 

    Parameters: 
    df (pd.DataFrame): DataFrame with sentiment scores 
    column (str): Name of the column that has the text 
    keyCol (str): Column with the keywords in the df 

    Returns:
    pd.DataFrame: Summary of the statistics for the keywords 
    """
    
    #Groupby the keywords and calculate statistics for the specific columns 
    key_sent = df.groupby(keyCol).agg({
        'sent_comp': ['count', 'std', 'mean'],
        'sent_pos': 'mean', 
        'sent_neg': 'mean', 
        'sent_type': lambda x: x.value_counts().to_dict()}).round(3)

    #Flatten the column names
    key_sent.columns = ['_'.join(col).strip() for col in key_sent.columns.values]
    key_sent = key_sent.reset_index()

    #Sort by the average compound sentiment score 
    key_sent = key_sent.sort_values('sent_comp_mean', ascending = False)

    #Rename sent_ID
    key_sent = key_sent.rename(columns = {'sent_type_<lambda>': "typeCount_summ"})

    return key_sent

def sent_result_graphs(df: pd.DataFrame, key_summ: pd.DataFrame) -> None:
    """
    Creates three graphs based on the findings from sentiment analysis. 
        1. Graph of the distribution of scores 
        2. Distribution of sentiment types 
        3. Average sentiment by keyword

    Parameters: 
    df (pd.DataFrame): DataFrame with sentiment scores 
    key_summ (pd.DataFrame): Dataframe with sentiment statistics of each keyword
    """

    #Set up
    fig, ax = plt.subplots(1,3, figsize = (25,5))

    #Graph 1 
    ax[0].hist(df['sent_comp'], bins = 30, color = 'pink', edgecolor = 'gray')
    ax[0].set_title('Distribution of Sentiment Scores')
    ax[0].set_xlabel('Compound Sentiment Score')
    ax[0].set_ylabel('Frequency')

    #Graph 2 
    sent_counts = df['sent_type'].value_counts()
    ax[1].pie(sent_counts.values, labels = sent_counts.index, autopct='%1.1f%%',
                colors = ['lightcoral', 'lightblue', 'lightgreen'])
    ax[1].set_title('Distribution of Sentiment Types')

    #Graph 3 
    keywords = key_summ['keyword']
    sent_means = key_summ['sent_comp_mean']

    bars = ax[2].barh(keywords, sent_means, color = 'lightseagreen')
    ax[2].set_title('Average Sentiment by Keyword')
    ax[2].set_xlabel('Average Compound Sentiment Score')
    ax[2].axvline(x = 0, color = 'blue', linestyle = '--')

    for bar, value in zip(bars, sent_means):
        ax[2].text(value, bar.get_y() + bar.get_height()/2, 
                     f'{value:.2f}', ha = "left", va = "center")

    plt.tight_layout()
    plt.show()


#Unit Testing
    if __name__ == "__main__":
    import unittest
    import warnings
    warnings.filterwarnings('ignore')
    unittest.main(argv=[''],exit=False)

class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        # Test data with known sentiment patterns
        self.test_data = pd.DataFrame({
            'byline': [
                "Eco-Friendly products are better for the environment!", # Positive
                "Pollution from excessive consumption is harming the environment.", # Negative
                "The Earth is round.",  # Neutral
                "", # Empty string
                "Mixed feelings on electric cars. It's good but can also be bad.",  # Mixed
                "12345", # Numbers only
                None # None value
            ],
            'keyword': ['product', 'product', 'cars', 'cars', 'product', 'cars', 'product'],
            'article_id': [1, 2, 3, 4, 5, 6, 7]
        })
        # Run sentiment analysis on test data
        self.df_with_sentiment = text_sentimentAnalysis(self.test_data, 'byline')
    def test_text_sentimentAnalysis_returns_dataframe(self):
        #Test that function returns a DataFrame
        result = text_sentimentAnalysis(self.test_data)
        self.assertIsInstance(result, pd.DataFrame)
        
    def test_text_sentimentAnalysis_adds_columns(self):
        #Test that all sentiment columns are added
        result = text_sentimentAnalysis(self.test_data)
        expected_columns = ['sent_scores', 'sent_comp', 'sent_pos', 'sent_neg', 'sent_neu', 'sent_type']
            
        for col in expected_columns:
            self.assertIn(col, result.columns, f"Column {col} missing from result")
        
    def test_text_sentimentAnalysis_handles_empty_strings(self):
        #Test that empty strings don't break the analysis
        empty_df = pd.DataFrame({'byline': ['', '', '']})
        result = text_sentimentAnalysis(empty_df)
        self.assertIn('sent_comp', result.columns)
        self.assertEqual(len(result), 3)
        
    def test_text_sentimentAnalysis_handles_none_values(self):
        #Test that None values are handled
        none_df = pd.DataFrame({'byline': [None, 'test', None]})
        result = text_sentimentAnalysis(none_df)
        self.assertIn('sent_comp', result.columns)
        self.assertEqual(len(result), 3)
        
    def test_text_sentimentAnalysis_sentiment_classification(self):
        #Test that sentiment types are correctly classified
        result = self.df_with_sentiment
            
        # Positive = 'pos'
        pos_text = "I love this amazing idea!"
        pos_df = pd.DataFrame({'byline': [pos_text]})
        pos_result = text_sentimentAnalysis(pos_df)
        self.assertEqual(pos_result['sent_type'].iloc[0], 'pos')
            
        # Negative text = 'neg'
        neg_text = "I hate this terrible idea!"
        neg_df = pd.DataFrame({'byline': [neg_text]})
        neg_result = text_sentimentAnalysis(neg_df)
        self.assertEqual(neg_result['sent_type'].iloc[0], 'neg')
            
        # Neutral text = 'neu'
        neu_text = "This is an idea."
        neu_df = pd.DataFrame({'byline': [neu_text]})
        neu_result = text_sentimentAnalysis(neu_df)
        self.assertEqual(neu_result['sent_type'].iloc[0], 'neu')
        
    def test_text_sentimentAnalysis_custom_column_name(self):
        #Test that function works with custom column names
        custom_df = pd.DataFrame({
            'custom_text': ['Good text', 'Bad text', 'Neutral text'],
            'other_col': [1, 2, 3]
        })
            
        result = text_sentimentAnalysis(custom_df, column='custom_text')
        self.assertIn('sent_comp', result.columns)
        self.assertEqual(len(result), 3)
        
    def test_key_sentimentAnalysis_returns_dataframe(self):
        #Test that key_sentimentAnalysis returns a DataFrame
        result = key_sentimentAnalysis(self.df_with_sentiment)
        self.assertIsInstance(result, pd.DataFrame)

    def test_key_sentimentAnalysis_statistics_calculated(self):
        #Test that statistics are calculated correctly
        result = key_sentimentAnalysis(self.df_with_sentiment)
            
        # For each keyword, check statistics exist
        for idx, row in result.iterrows():
            self.assertIsNotNone(row['sent_comp_count'])
            self.assertIsNotNone(row['sent_comp_mean'])
            self.assertIsNotNone(row['sent_pos_mean'])
            self.assertIsNotNone(row['sent_neg_mean'])
            self.assertIsInstance(row['typeCount_summ'], dict)
    def test_key_sentimentAnalysis_sorting(self):
        #Test that results are sorted by sent_comp_mean in descending order
        result = key_sentimentAnalysis(self.df_with_sentiment)
            
        # Check if in descending order
        comp_means = result['sent_comp_mean'].tolist()
        self.assertEqual(comp_means, sorted(comp_means, reverse=True),
                         "DataFrame should be sorted by sent_comp_mean descending")
