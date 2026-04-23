
# Fake News Detection Using NLP
import pandas as pd
import string, re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')

fake = pd.read_csv('data/fake.csv')
true = pd.read_csv('data/true.csv')

fake['label'] = 0
true['label'] = 1

df = pd.concat([fake, true], ignore_index=True)

print(df['label'].value_counts())

df = df.sample(frac=1).reset_index(drop=True)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['content'] = df['title'] + " " + df['text']
df['clean_text'] = df['content'].apply(clean_text)


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

while True:
    news = input("\nEnter news to check (or type exit): ")
    if news.lower() == "exit":
        break
    
    clean = clean_text(news)
    vector = vectorizer.transform([clean])
    result = model.predict(vector)

    if result[0] == 1:
        print("✅ Real News")
    else:
        print("❌ Fake News")

