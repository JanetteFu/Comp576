import pandas as pd
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

files.upload()

steam_user = pd.read_csv("user_info.csv",header=None)
steam_user.rename(columns={0:"id",1:"game",2:"action",3:"hour"}, inplace=True)

steam_user.head()

steam_purchase = steam_user[steam_user["action"] == "purchase"]

observed = []
target = []

files.upload()

steam_game = pd.read_csv("game_info.csv")

steam_game.head()

import re

# Replace "Free", "Free to Play" and handle NaN with "$0.00"
steam_game['original_price'] = steam_game['original_price'].fillna("$0.00")

def clean_price(price):
    # Check if the price doesn't match the format "$ XX.XX"
    if not re.match(r'^\$\d+(\.\d{2})?$', str(price)):
        return "$0.00"
    else:
        return price

steam_game['original_price'] = steam_game['original_price'].apply(clean_price)
steam_game['original_price'] = steam_game['original_price'].replace({'\$': '', ',': ''}, regex=True).astype(float)

steam_game['original_price'].describe()

steam_game.rename(columns={'name':'game'}, inplace=True)

steam_final = pd.merge(steam_purchase, steam_game, how="left", on="game")
steam_final.head()

steam_final = steam_final.dropna(subset=["genre"])
steam_final.describe()

user_counts = steam_final['id'].value_counts()
filtered_ids = user_counts[user_counts > 9].index

# Select rows corresponding to those IDs
final_users = steam_final[steam_final['id'].isin(filtered_ids)]
final_users.to_csv('/content/final_users.csv', index=False)

final_users = pd.read_csv("final_users.csv")
final_users.head()

for user_id, group in final_users.groupby('id'):
    if len(group) >= 6:
        # Sort and extract the first 5 as observed and the 6th as the target
        group_sorted = group.head(6)
        observed.append(group_sorted.iloc[:5].reset_index(drop=True))
        target.append(group_sorted.iloc[5:6].reset_index(drop=True))

# Combine all observed and target users into DataFrames
observed_user = pd.concat(observed, ignore_index=True)
target_user = pd.concat(target, ignore_index=True)

observed_user['game'].describe()

game_counts = observed_user['game'].value_counts()
top_10_games = game_counts.head(10)

# Plot the frequency as a bar plot
plt.figure(figsize=(10, 6))
top_10_games.plot(kind='bar', color='skyblue')

# Adding labels and title
plt.title('Top 10 Frequency of Games', fontsize=16)
plt.xlabel('Game', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=90)  # Rotate game names for better readability
plt.tight_layout() 

# Show the plot
plt.show()
plt.savefig('top_10_games_frequency.png', dpi=300)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
observed_user['languages'] = observed_user['languages'].fillna("None")
languages = vectorizer.fit_transform(observed_user['languages'])

# Apply TF-IDF Transformation to the tokenized data
transformer = TfidfTransformer()
languages_tfidf = transformer.fit_transform(languages)

languages_tfidf = pd.DataFrame(languages_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

observed_user['genre'] = observed_user['genre'].fillna("None")
genre = vectorizer.fit_transform(observed_user['genre'])

transformer = TfidfTransformer()
genre_tfidf = transformer.fit_transform(genre)

genre_tfidf = pd.DataFrame(genre_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Count word frequencies
split_genres = [sentence.split(", ") for sentence in observed_user['genre']]

flat_genres = [genre for sublist in split_genres for genre in sublist]
word_counts = Counter(flat_genres)

# word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis'
).generate_from_frequencies(word_counts)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most popular combination of genres", fontsize=16)
plt.show()

import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

#tokenize and embedding using bert
def get_bert_embeddings_batch(text_list):

    processed_texts = []
    for text in text_list:
        if not isinstance(text, str) or text.strip() == "":  # Handle empty or invalid text
            processed_texts.append("[EMPTY]")  # Default placeholder
        else:
            # Truncate text intelligently
            truncated_text = text[:512] if len(text) > 512 else text
            processed_texts.append(truncated_text)
    
    # Tokenize and process batch
    inputs = tokenizer(processed_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Compute mean pooling of the embeddings
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def store_embeddings_in_dataframe(df, column_name, embedding_function):
    embeddings = embedding_function(df[column_name].tolist())
    df[f'{column_name}_bert'] = [list(embed) for embed in embeddings]  # Ensure each embedding is stored as a list

# Apply the embedding function to each relevant column
store_embeddings_in_dataframe(observed_user, 'game_description', get_bert_embeddings_batch)
store_embeddings_in_dataframe(observed_user, 'game_details', get_bert_embeddings_batch)
store_embeddings_in_dataframe(observed_user, 'popular_tags', get_bert_embeddings_batch)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
"""PCA for dim-reduction"""
pca = PCA(n_components=15)
description_embedding = pca.fit_transform(description_embedding)
details_embedding = pca.fit_transform(description_embedding)
pca = PCA(n_components=5)
tags_embedding = pca.fit_transform(tags_embedding)

tags_embedding_df = pd.DataFrame(tags_embedding, columns=[f'pca_{i+1}' for i in range(5)])
details_embedding_df = pd.DataFrame(details_embedding, columns=[f'pca_{i+1}' for i in range(15)])
description_embedding_df = pd.DataFrame(description_embedding, columns=[f'pca_{i+1}' for i in range(15)])

observed_result = observed_user[['id','game']]
previous_cleanedText = pd.concat([observed_user['id'],observed_user['original_price'], description_embedding_df,
                               genre_tfidf, languages_tfidf, details_embedding_df, tags_embedding_df], axis=1)   
previous_cleanedText.head()

previous_cleanedText['times'] = previous_cleanedText.groupby('id').cumcount() + 1
previous_cleanedText_1 = previous_cleanedText[previous_cleanedText['times'] == 1]
previous_cleanedText_2 = previous_cleanedText[previous_cleanedText['times'] == 2]
previous_cleanedText_3 = previous_cleanedText[previous_cleanedText['times'] == 3]
previous_cleanedText_4 = previous_cleanedText[previous_cleanedText['times'] == 4]
previous_cleanedText_5 = previous_cleanedText[previous_cleanedText['times'] == 5]
previous_cleanedText_6 = previous_cleanedText[previous_cleanedText['times'] == 6]
previous_cleanedText_7 = previous_cleanedText[previous_cleanedText['times'] == 7]
previous_cleanedText_8 = previous_cleanedText[previous_cleanedText['times'] == 8]
previous_cleanedText_9 = previous_cleanedText[previous_cleanedText['times'] == 9]

# Concatenate the two DataFrames vertically (i.e., stacking them along the columns)
grouped_cleanedText = pd.concat([previous_cleanedText_1.drop(columns=['times']).reset_index(drop=True), previous_cleanedText_2.drop(columns=['id','times']).reset_index(drop=True), previous_cleanedText_3.drop(columns=['id','times']).reset_index(drop=True), previous_cleanedText_4.drop(columns=['id','times']).reset_index(drop=True), 
                                 previous_cleanedText_5.drop(columns=['id','times']).reset_index(drop=True),previous_cleanedText_6.drop(columns=['id','times']).reset_index(drop=True), previous_cleanedText_7.drop(columns=['id','times']).reset_index(drop=True), previous_cleanedText_8.drop(columns=['id','times']).reset_index(drop=True), 
                                 previous_cleanedText_9.drop(columns=['id','times']).reset_index(drop=True)], axis=1)

observed_target = observed_result.groupby('id')['game'].apply(', '.join).reset_index(name='concatenated_games')

final_target = target_user['game'] 

"""Build the KNN model"""
def knn_evaluate(selected, observed_target, final_target):
    correct_select_5 = [0]*len(final_target)
    correct_select_15 = [0]*len(final_target)
    for i in range(len(final_target)):
      temp_selected = np.argsort(-selected[i])[:7]
      for j in selected:
        split_string = observed_target[j].split()
        select_game = np.concatenate(split_string)
      word_counts = Counter(select_game)

      sorted_games = [game for game, _ in word_counts.most_common()]

      select_5 = sorted_games[:5]
      select_15 = sorted_games[:15]
      if(final_target[i] in select_5):
          correct_select_5[i] = 1
      if(final_target[i] in select_15):
          correct_select_15[i] = 1

    accuracy_5 = sum(correct_select_5) / len(final_target)
    accuracy_15 = sum(correct_select_15) / len(final_target)
    return accuracy_5, accuracy_15

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(pca_cleanedText, final_target, test_size=0.3, random_state=42)

k_values = range(3, 12)
accuracies = []

# Loop through K values
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Perform cross-validation
    knn.fit(X_train, y_train)
    selected = knn.predict_proba(X_test)
    scores = knn_evaluate(selected, observed_target['game'], final_target)

    # Append mean accuracy
    accuracies.append(scores)

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', color='blue', linewidth=2)
plt.xlabel('Number of Neighbors (K)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(k_values, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()

"""Build the LSTM model"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical

# Prepare the data
X = pca_cleanedText
y = to_categorical(np.array(final_target))  # Replace with one-hot encoded target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vocab_size = X_train.shape[1]
# Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=X.shape[1]))  # Adjust vocab_size and input_length
model.add(LSTM(64, return_sequences=False))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
LSTM_predicted_probabilities = model.predict(X_test)

"""Combine two predictions"""

def ensemble_evaluate(selected, observed_target, game_list, final_target, predicted_probabilities):
    top_5_games = []
    top_15_games = []
    for i in range(len(final_target)):
    # For LSTM
      nn_sorted_indices = np.argsort(-predicted_probabilities)
      nn_sorted_games = [game_list[i] for i in nn_sorted_indices]
      nn_ranks = {game: rank for rank, game in enumerate(nn_sorted_games, start=1)}

    # For KNN
      temp_selected = np.argsort(-selected[i])[:7]
      for j in selected:
        split_string = observed_target[j].split()
        knn_select_game = np.concatenate(split_string)
      knn_word_counts = Counter(knn_select_game)
      knn_sorted_games = [game for game, _ in knn_word_counts.most_common()]
      knn_ranks = {game: rank for rank, game in enumerate(knn_sorted_games, start=1)}

    # Combine the ranks using equal weights
      combined_scores = {}
      all_games = set(nn_ranks.keys()).union(knn_ranks.keys())
      for game in all_games:
          nn_rank = nn_ranks.get(game, len(nn_sorted_games) + 1)  # Default rank for unseen games
          knn_rank = knn_ranks.get(game, len(knn_sorted_games) + 1)  # Default rank for unseen games
          combined_scores[game] = 0.5 * nn_rank + 0.5 * knn_rank

    # Sort games by combined scores
      sorted_combined_games = sorted(combined_scores.items(), key=lambda x: x[1])

    # Extract the top-5 and top-15 games
      top_5_games.append([game for game, score in sorted_combined_games[:5]])
      top_15_games.append([game for game, score in sorted_combined_games[:15]])

    return top_5_games, top_15_games

def accuracy_ensemble(top_5_games, top_15_games, final_target):
    correct_select_5 = [0]*len(final_target)
    correct_select_15 = [0]*len(final_target)
    for i in range(len(final_target)):
      if(final_target[i] in top_5_games[i]):
          correct_select_5[i] = 1
      if(final_target[i] in top_15_games[i]):
          correct_select_15[i] = 1

    accuracy_5 = sum(correct_select_5) / len(final_target)
    accuracy_15 = sum(correct_select_15) / len(final_target)
    return accuracy_5, accuracy_15

top_5_games, top_15_games = ensemble_evaluate(selected, observed_target, y_train, final_target, LSTM_predicted_probabilities)
ensemble_accuracy = accuracy_ensemble(top_5_games, top_15_games, final_target)

"""Logistic Baseline"""

def accuracy_logistic(game_list, final_target, predicted_probabilities):
    correct_5 = [0]*len(final_target)
    correct_15 = [0]*len(final_target)
    for i in range(len(final_target)):
      log_sorted_indices = np.argsort(-predicted_probabilities[i])
      log_sorted_games = [game_list[i] for i in log_sorted_indices]

    # Extract the top-5 and top-15 games
      top_5_games = [game for game, score in log_sorted_games[:5]]
      top_15_games = [game for game, score in log_sorted_games[:15]]

      if(final_target[i] in top_5_games):
        correct_5[i] = 1
      if(final_target[i] in top_15_games):
        correct_15[i] = 1

    accuracy_5 = sum(correct_5) / len(final_target)
    accuracy_15 = sum(correct_15) / len(final_target)
    return accuracy_5, accuracy_15

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(pca_cleanedText, final_target, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict probabilities for all games in the dataset
log_predicted_probabilities_log = model.predict_proba(scaler.transform(X))

log_accuracy_5, log_prediction_accuracy_15 = accuracy_logistic(y_train, final_target, log_predicted_probabilities)

"""SVM for comparison"""

from sklearn.svm import SVC

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(pca_cleanedText, final_target, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model with probability estimation
svm_model = SVC(kernel='linear', probability=True, random_state=42)  # You can also try 'rbf' kernel
svm_model.fit(X_train_scaled, y_train)

# Predict probabilities for all games in the dataset
SVM_predicted_probabilities = svm_model.predict_proba(scaler.transform(X))[:, 1]
SVM_accuracy_5, SVM_accuracy_15 = accuracy_logistic(y_train, final_target, SVM_predicted_probabilities)
