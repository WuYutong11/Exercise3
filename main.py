import nltk
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import string

# Download NLTK resources if not already done
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('gutenberg')

# Read Moby Dick from the Gutenberg dataset
moby_dick = gutenberg.raw('melville-moby_dick.txt')

# Tokenization
tokens = word_tokenize(moby_dick)

# Stop-words filtering
stop_words = set(stopwords.words('english'))
filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word not in string.punctuation]

# Parts-of-Speech (POS) tagging
pos_tags = pos_tag(filtered_tokens)

# POS frequency
fdist = FreqDist(tag for word, tag in pos_tags)
common_pos = fdist.most_common(5)

# Lemmatization
lemmatizer = WordNetLemmatizer()
top_20_tokens = [word for word, _ in FreqDist(filtered_tokens).most_common(20)]
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in top_20_tokens]

# Plotting frequency distribution
labels, values = zip(*common_pos)
plt.bar(labels, values)
plt.xlabel('Parts of Speech')
plt.ylabel('Frequency')
plt.title('POS Frequency Distribution')
plt.show()

# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = [analyzer.polarity_scores(sentence)['compound'] for sentence in nltk.sent_tokenize(moby_dick)]
average_sentiment = sum(sentiment_scores) / len(sentiment_scores)

if average_sentiment > 0.05:
    sentiment = 'positive'
else:
    sentiment = 'negative'

print(f"Average Sentiment Score: {average_sentiment}")
print(f"Overall Text Sentiment: {sentiment}")
