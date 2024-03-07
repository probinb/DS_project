import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import emoji
import demoji
import nltk
from nltk.corpus import stopwords

# Load the preprocessed data
df = pd.read_csv('/home/ubuntu/Streamlit/Supply_Chain_preprocessed.csv')

st.title("Supply Chain : Sentiment Analysis Project")
st.sidebar.title("Table of contents")
pages=["Our Project","Data Processing","Wordclouds", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)


if page == pages[0] : 
    st.write("### Our Project")
    st.write(
    "Our project is driven by the aim of extracting valuable insights from product reviews, uncovering the core strengths and weaknesses. Our exploration is guided by the following questions:\n\n"
    "1. **Sentiment Classification:** Identifying a robust classification algorithm to predict the sentiment of reviews as positive, negative, or neutral.\n"
    "2. **Thematic Analysis:** Uncovering themes that contribute to the positive, negative, or neutral classification of reviews, providing deeper insights into customer sentiments.\n"
    "3. **Brand Comparison:** Investigating whether issues highlighted in reviews vary among products from major brands, shedding light on brand-specific considerations.\n"
    "4. **Top Voted Reviews:** Identifying significant issues found in reviews with the highest number of votes, highlighting key concerns that resonate with a larger audience.\n\n"
    "Through our Streamlit app, you'll have the opportunity to explore these objectives, visualize the data, and gain valuable insights into the dynamics of customer reviews."
)

    st.write("Our data science project focuses on the analysis of online product reviews to provide valuable insights into consumer purchasing decisions.",
             "The project aims to use machine learning algorithms to extract crucial information from a dataset obtained from Kaggle, specifically focusing on cellphones.")
    st.dataframe(df.head(30))


if page == pages[1] : 
    st.write("### Focus of Dataset")
    
    st.write("To ensure a more targeted analysis, we narrowed our data down to cellphones only.",
         "This refinement allows us to extract insights directly relevant to the supply chain of mobile devices,",
         "eliminating unnecessary noise from cellphone accessories -therefor enhancing the significance of our findings.\n")

    st.write("### Dataprocessing")
    st.write("The initial dataset faced challenges with duplicated rows and missing values.",
         "We addressed this by removing unnecessary duplicates and deleting entries without reviews.",
         "For a large amount of unknown Brandnames, we traced their brandname from the Product Name."
         "We reduced inconsistent capitalization and formatting in the Brand Name column.",
         "Furthermore we deleted all cellphone accessories contained in the dataset to focus only on the cellphones.")

    st.write(
        "Data preprocessing plays a crucial role in constructing effective Machine Learning models, "
        "as the quality of results is closely tied to the thoroughness of data preprocessing. "
        "Our text preprocessing pipeline involved several key steps:\n\n"
        "- **Emoji Handling:** Replacing emojis using the vocabulary from the demoji package, ensuring "
        "that emoticons are appropriately represented in the text.\n"

        "- **Digit Removal:** Eliminating numerical digits from the text, focusing on the textual content "
        "rather than numerical values.\n"
        "- **Exclamation Mark Replacement:** Substituting exclamation marks with the word 'exclamation' "
        "to capture their emotional significance in the text.\n"
        "- **Special Character Removal:** Removing other special characters to maintain a clean and "
        "standardized text format.\n"
        "- **Lowercasing:** Converting all words to lowercase to ensure uniformity and prevent "
        "discrepancies due to case variations.\n"
        "- **Stopword Removal:** Eliminating common stopwords, which were augmented with domain-specific "
        "terms from our reviews to enhance the relevance of the stopword list.\n"
        "- **Tokenization:** Splitting sentences into individual words (tokenization), a crucial step "
        "for further analysis and feature extraction.\n\n")

    st.write("#### An Example")
    import demoji
    import re
    import emoji
    import nltk
    from nltk.corpus import stopwords

    # Emoji Handling
    def replace_emojis(text):
        return ' '.join([demoji.replace_with_desc(emo) for emo in text.split()])

    # Digit Removal
    def remove_digits(text):
        return re.sub(r"\d+", "", text)

    # Exclamation Mark Replacement
    def replace_exclamation(text):
        return re.sub(r"!", " exclamation ", text)

    # Special Character Removal
    def remove_special_characters(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Lowercasing
    def convert_to_lowercase(text):
        return text.lower()

    # Stopword Removal
    nltk.download('stopwords')
    nltk.download('punkt')
    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        custom_stopwords = ['phone', 'phones', 'cell', 'amazon', 'review', 'reviews', 'product', 'products',
                            'buy', 'samsung', 'apple', 'android', 'mobile', 'galaxy', 'google',
                            'iphone', 'verizon', 'work', 'movistar', 'telefonica', 'lg', 'htc', 'maybe', 'blu',
                            'really', 'very', 'new', 'work', 'get', 'say']
        stop_words.update(custom_stopwords)
        
        return ' '.join([word for word in text.split() if word not in stop_words])

    # Tokenization
    def tokenize_text(text):
        return nltk.word_tokenize(text)

    # Example usage
    original_text = "This is a sample text with 😊, 5 stars, and some special characters! It's great!"
    processed_text = replace_emojis(original_text)
    processed_text = remove_digits(processed_text)
    processed_text = replace_exclamation(processed_text)
    processed_text = remove_special_characters(processed_text)
    processed_text = convert_to_lowercase(processed_text)
    processed_text = remove_stopwords(processed_text)
    tokenized_text = tokenize_text(processed_text)

    # Display processed text and tokenized words
    st.write("Example for an original Text:", original_text)
    st.write("Processed Text of the Example:", processed_text)
    st.write("Tokenized Words of the given example:", tokenized_text)

if page == pages[1] :
    st.write("### DataVizualization")
  
if page == pages[2] : 
    st.write("### SUPPLY CHAIN IMPLICATIONS")
'''
# Separate df into positive, neutral and negative data
df_neg = df[df["sentiment"] == "Negative"]
df_pos = df[df["sentiment"] == "Positive"]
df_neutral = df[df["sentiment"] == "Neutral"]

import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from nltk.tokenize import word_tokenize

tokens_pos = word_tokenize(pos_reviews, language='english')
tokens_neg = word_tokenize(neg_reviews, language='english')
tokens_neutral = word_tokenize(neutral_reviews, language='english')

# Function to generate word clouds
def generate_word_cloud(reviews, title, mask_image):
    text = ' '.join(reviews)
    wc = WordCloud(
        background_color="white",
        max_words=1000,
        colormap='Greens' if 'positive' in title.lower() else 'Reds',
        stopwords=stop_words,
        max_font_size=400,
        collocations=False,
        random_state=42,
        mask=mask_image
    )
    wc.generate(text)

    # Display the word cloud
    st.title(title)
    st.image(wc.to_image())

# Importing Images for masks
mask_pos = np.array(Image.open("/home/ubuntu/Streamlit/jpegs/heart.jpg"))
mask_neg = np.array(Image.open("/home/ubuntu/Streamlit/jpegs/dislike.jpg"))
mask_neutral = np.array(Image.open("/home/ubuntu/Streamlit/jpegs/neutral.jpg"))

# Example usage in Streamlit app
st.title("Word Clouds")
st.header("Positive Reviews")
generate_word_cloud(pos_reviews, "Positive Reviews Word Cloud", mask_pos)

st.header("Negative Reviews")
generate_word_cloud(neg_reviews, "Negative Reviews Word Cloud", mask_neg)

st.header("Negative Reviews")
generate_word_cloud(neutral_reviews, "Negative Reviews Word Cloud", mask_neutral)


# Display Word Clouds for Samsung and Apple
st.header("Brand and Sentiment Selection")
selected_brand = st.selectbox("Select Brand", ["SAMSUNG", "APPLE"])
selected_sentiment = st.selectbox("Select Sentiment", ["Negative", "Positive", "Neutral"])

# Filter Data based on Brand and Sentiment
filtered_data = df[(df['Brand Name'] == selected_brand) & (df['sentiment'] == selected_sentiment)]

# Display Basic Stats and Word Clouds
st.header("Basic Statistics")
st.write(f"Total Reviews for {selected_brand} with {selected_sentiment} sentiment: {len(filtered_data)}")
st.write(f"Average Rating: {filtered_data['Stars'].mean():.2f}")
st.write(f"Average Word Count: {filtered_data['Word Count'].mean():.2f}")

# Display Word Cloud for Selected Brand and Sentiment
generate_word_cloud(filtered_data['Text'], f"Word Cloud for {selected_brand} {selected_sentiment} Reviews")

# Display Reviews
st.header("Reviews")
st.dataframe(filtered_data[['Text', 'Stars', 'sentiment']])

# Distribution of Review Lengths
st.header("Distribution of Review Lengths")
sns.set_theme(font_scale=1)
sns_plot = sns.boxplot(x="Word Count", data=filtered_data, palette="Set3")
st.pyplot(sns_plot.figure)

# Dataframes separating Samsung, Apple, and BLU reviews by sentiments
bad_rating_samsung = df[(df['sentiment'] == 'Negative') & (df['Brand Name'] == 'SAMSUNG')]
good_rating_samsung = df[(df['sentiment'] == 'Positive') & (df['Brand Name'] == 'SAMSUNG')]
middle_rating_samsung = df[(df['sentiment'] == 'Neutral') & (df['Brand Name'] == 'SAMSUNG')]

bad_rating_apple = df[(df['sentiment'] == 'Negative') & (df['Brand Name'] == 'APPLE')]
good_rating_apple = df[(df['sentiment'] == 'Positive') & (df['Brand Name'] == 'APPLE')]
middle_rating_apple = df[(df['sentiment'] == 'Neutral') & (df['Brand Name'] == 'APPLE')]

bad_rating_blu = df[(df['sentiment'] == 'Negative') & (df['Brand Name'] == 'BLU')]
good_rating_blu = df[(df['sentiment'] == 'Positive') & (df['Brand Name'] == 'BLU')]
middle_rating_blu = df[(df['sentiment'] == 'Neutral') & (df['Brand Name'] == 'BLU')]


# Display options for brand and sentiment
selected_brand = st.selectbox("Select Brand", ["SAMSUNG", "APPLE", "BLU"])
selected_sentiment = st.selectbox("Select Sentiment", ["Negative", "Positive", "Neutral"])

# Display selected DataFrame based on brand and sentiment
if selected_brand == "SAMSUNG":
    if selected_sentiment == "Negative":
        st.dataframe(bad_rating_samsung)
    elif selected_sentiment == "Positive":
        st.dataframe(good_rating_samsung)
    elif selected_sentiment == "Neutral":
        st.dataframe(middle_rating_samsung)
        
elif selected_brand == "APPLE":
    if selected_sentiment == "Negative":
        st.dataframe(bad_rating_apple)
    elif selected_sentiment == "Positive":
        st.dataframe(good_rating_apple)
    elif selected_sentiment == "Neutral":
        st.dataframe(middle_rating_apple)
        
elif selected_brand == "BLU":
    if selected_sentiment == "Negative":
        st.dataframe(bad_rating_blu)
    elif selected_sentiment == "Positive":
        st.dataframe(good_rating_blu)
    elif selected_sentiment == "Neutral":
        st.dataframe(middle_rating_blu)
'''
if page == pages[4] :
    st.write("### Modelling")
    st.write("X contains the text data (reviews), and y contains the corresponding sentiment labels (positive, negative, or neutral).",
    "The data is split into a training and test set using the train_test_split function from scikit-learn.",
    "It allocates 75percent of the data to training and 25% to testing. We use the CountVectorizer to convert the text data into numerical vectors.",
    "It represents each single review as a bag-of-words, counting the occurrences of each word")
    
    import pickle
    import joblib
    from sklearn.metrics import confusion_matrix

    options = ['Random Forest', 'Decision Tree', 'Logistic Regression', 'Gradient Boosting']
    option = st.selectbox('Choice of the model', options)
    st.write('The chosen model is:', option)
    
    # Load the trained model using Joblib
    model_filename = f"{option.lower().replace(' ', '_')}_model.joblib"
    clf = joblib.load(model_filename)

    # Calculate accuracy
    accuracy_filename = f"{option.lower().replace(' ', '_')}_accuracy.pkl"
    with open(accuracy_filename, 'rb') as accuracy_file:
        accuracy = pickle.load(accuracy_file)

    # Calculate confusion matrix
    confusion_matrix_filename = f"{option.lower().replace(' ', '_')}_confusion_matrix.pkl"
    with open(confusion_matrix_filename, 'rb') as confusion_matrix_file:
        confusion_matrix_result = pickle.load(confusion_matrix_file)

    display = st.radio('What do you want to show?', ('Accuracy', 'Confusion matrix'))
    # Display the chosen information
    if display == 'Accuracy':
        st.write(f'Accuracy for {option}: {accuracy}')
    elif display == 'Confusion matrix':
        st.dataframe(confusion_matrix_result)