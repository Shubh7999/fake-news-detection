import pandas as pd
import pickle

print("Step 1: Loading dataset...")

# Load dataset
import os

# Get current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build correct dataset paths
fake_path = os.path.join(current_dir, '..', 'data', 'Fake.csv')
true_path = os.path.join(current_dir, '..', 'data', 'True.csv')

# Load dataset
fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

fake['label'] = 0
true['label'] = 1

df = pd.concat([fake.head(1000), true.head(1000)])
df = df.sample(frac=1)

print("Dataset loaded successfully")
print("Total rows:", len(df))

X = df['text']
y = df['label']

print("Step 2: Tokenizing text...")

# NLP
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100)

print("Tokenization completed")

print("Step 3: Building CNN model...")

# CNN MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense

model = Sequential([
    Embedding(5000, 64, input_length=100),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(),
    Flatten(),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Step 4: Training started...")

model.fit(X_pad, y, epochs=3, batch_size=32, verbose=1)

print("Step 5: Saving model...")

model.save('cnn_model.h5')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("✅ CNN Model trained successfully")