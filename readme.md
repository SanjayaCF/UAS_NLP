# Text-Analysis-and-Prediction-App

This project is a web-based application that analyzes a corpus of text documents. It provides functionalities for text preprocessing, n-gram analysis, word prediction, and relevant document retrieval using TF-IDF and cosine similarity.

## Team Members

* **71220840** - Rendy Ananta Kristanto
* **71220841** - Yosua Sutanto Putra
* **71220965** - Sanjaya Cahyadi Fuad

---

## Features

* **Document Upload:** Users can upload multiple .txt documents for processing.
* **Corpus Statistics:** Displays statistics of the processed documents, including the total number of documents, and the counts of unigrams, bigrams, and trigrams.
* **Word Cloud and Top Tokens:** Generates a word cloud and a bar chart of the top 10 most frequent tokens in the corpus.
* **Word Prediction:** Suggests the next word based on a given query of one or two words, using a trigram model with Laplace smoothing.
* **Relevant Document Search:** Retrieves and ranks the most relevant documents for a given query using TF-IDF and cosine similarity.
* **Reset Functionality:** Allows users to reset the entire corpus, deleting all uploaded documents and saved states.

---

## How It Works

The application is built with **Flask**, a Python web framework. Here's a breakdown of the core components:

### 1. Preprocessing

When a text file is uploaded, it goes through the following preprocessing steps:
1.  **Lowercase Conversion:** The entire text is converted to lowercase.
2.  **Punctuation and Special Character Removal:** All non-alphabetic characters are removed.
3.  **Tokenization:** The text is split into a list of words (tokens).
4.  **Stopword Removal:** Common Indonesian stopwords are removed using the `nltk` library.
5.  **Stemming:** Words are reduced to their root form using the `Sastrawi` library.

### 2. N-Gram Language Model

The application builds unigram, bigram, and trigram models from the processed tokens:
* **Unigram, Bigram, and Trigram Counts:** The frequencies of single tokens, pairs of consecutive tokens, and triplets of consecutive tokens are counted.
* **Word Prediction:** For a given input query, the application uses the trigram and bigram models to predict the next word. It applies **Laplace smoothing** to handle unseen n-grams and calculates the probability of each potential next word.

### 3. Document Retrieval (TF-IDF and Cosine Similarity)

To find relevant documents for a query, the application uses the **TF-IDF (Term Frequency-Inverse Document Frequency)** vector space model:
* **TF (Term Frequency):** The frequency of each term in a document.
* **IDF (Inverse Document Frequency):** The inverse of the frequency of a term across all documents. This gives higher weight to terms that are rare across the corpus.
* **TF-IDF Vector:** Each document is represented as a vector of TF-IDF scores for each term.
* **Cosine Similarity:** When a user enters a query, it is also converted into a TF-IDF vector. The **cosine similarity** is then calculated between the query vector and each document vector to determine the relevance of the documents. The documents are then ranked and displayed in descending order of their similarity scores.

### 4. Persistence

The state of the application, including the processed documents, n-gram counts, and TF-IDF vectors, is saved to a pickle file (`corpus_data.pkl`). This allows the application to be restarted without losing the processed data.

---

## How to Use

1.  **Run the application:**
    ```bash
    python app.py
    ```
2.  **Upload Documents:**
    * Click on the "Choose Files" button and select one or more `.txt` files to upload.
    * Click "Upload & Proses Dokumen" to process the files.
3.  **View Corpus Information:**
    * The "Total" section displays the overall statistics of the corpus.
    * The "Word Cloud" and "Top 10 Token" sections visualize the most frequent words.
4.  **Predict Next Word and Find Relevant Documents:**
    * Enter one or two words in the "Query" input field.
    * Click "Prediksi & Cari Dokumen".
    * The "Prediksi Kata Berikutnya" table shows the most likely next words with their probabilities.
    * The "Dokumen Relevan" table displays the documents most relevant to your query, along with a snippet of the text where the query appears.

---

## Formulas Used

### Term Frequency (TF)
This formula calculates how often a term appears in a document.
![Rumus TF](https://i.imgur.com/your_tf_image_url.png)

### Inverse Document Frequency (IDF)
This formula gives a higher weight to terms that are rare across all documents.
![Rumus IDF](https://i.imgur.com/your_idf_image_url.png)

### TF-IDF
This score is the product of TF and IDF and represents the importance of a term in a document relative to the entire corpus.
![Rumus TF-IDF](https://i.imgur.com/your_tfidf_image_url.png)

### Cosine Similarity
We use this to measure the similarity between the query and each document based on their TF-IDF vectors.
![Rumus Cosine Similarity](https://i.imgur.com/your_cosine_image_url.png)

### Laplace Smoothing
This is used in our N-gram model to handle words that haven't been seen before, preventing zero-probability issues.
![Rumus Laplace Smoothing](https://i.imgur.com/your_laplace_image_url.png)

---
## Project Structure

```

├── app.py \# Main Flask application
├── readme.md \# This file
├── DATASOURCES.md \# List of data sources
├── static
│   ├── css
│   │   └── style.css \# Stylesheet
│   ├── js
│   │   └── script.js \# JavaScript for interactivity
│   ├── top\_tokens.png \# Generated plot of top tokens
│   └── wordcloud.png \# Generated word cloud
├── templates
│   └── index.html \# Main HTML template
└── uploads \# Directory for uploaded text files
├── rendy1.txt
├── rendy2.txt
├── ...

```

---

## Data Sources

This project was developed using a corpus of articles from various online news and academic sources. For a complete list of the data sources, please see [DATASOURCES.md](DATASOURCES.md).
