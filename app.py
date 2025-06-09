import os
import re
import math
import pickle
from collections import Counter, defaultdict
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

PICKLE_PATH = 'corpus_data.pkl'

try:
    stop_words = set(stopwords.words('indonesian'))
except:
    stop_words = set()

factory = StemmerFactory()
stemmer = factory.create_stemmer()

documents = {}
doc_term_freq = {}

unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()
total_tokens = 0

df = defaultdict(int)
doc_tf_idf = {}

doc_count = 0


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = re.findall(r'\b[a-z]+\b', text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return tokens


def build_ngrams(tokens):
    global unigram_counts, bigram_counts, trigram_counts, total_tokens
    for i in range(len(tokens)):
        unigram_counts[tokens[i]] += 1
        total_tokens += 1
        if i < len(tokens) - 1:
            bigram_counts[(tokens[i], tokens[i+1])] += 1
        if i < len(tokens) - 2:
            trigram_counts[(tokens[i], tokens[i+1], tokens[i+2])] += 1


def recompute_df_and_tfidf():
    global df, doc_tf_idf
    df.clear()

    for tokens in documents.values():
        unique_terms = set(tokens)
        for t in unique_terms:
            df[t] += 1

    N = len(documents)
    doc_tf_idf.clear()
    for fname, term_counts in doc_term_freq.items():
        tfidf_vec = {}
        for term, tf in term_counts.items():
            idf = math.log((N) / df[term]) if df[term] > 0 else 0.0
            tfidf_vec[term] = tf * idf
        doc_tf_idf[fname] = tfidf_vec


def visualisasi():
    wc = WordCloud(width=800, height=400, background_color='white')\
            .generate_from_frequencies(unigram_counts)
    wc_path = os.path.join('static', 'wordcloud.png')
    wc.to_file(wc_path)

    top_tokens = unigram_counts.most_common(10)
    if top_tokens:
        words, freqs = zip(*top_tokens)
    else:
        words, freqs = [], []
    plt.figure(figsize=(8, 5))
    plt.bar(words, freqs)
    plt.xlabel('Token')
    plt.ylabel('Frekuensi')
    plt.title('Top 10 Token')
    plt.xticks(rotation=45)
    top_tokens_path = os.path.join('static', 'top_tokens.png')
    plt.tight_layout()
    plt.savefig(top_tokens_path)
    plt.close()


def cosine_similarity(vec1, vec2):
    dot = 0.0
    for term, v in vec1.items():
        dot += v * vec2.get(term, 0.0)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def save_state_to_pickle():
    state = {
        'documents': documents,
        'doc_term_freq': doc_term_freq,
        'df': df,
        'doc_tf_idf': doc_tf_idf,
        'unigram_counts': unigram_counts,
        'bigram_counts': bigram_counts,
        'trigram_counts': trigram_counts,
        'total_tokens': total_tokens,
        'doc_count': doc_count
    }
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(state, f)


def load_state_from_pickle():
    global documents, doc_term_freq, df, doc_tf_idf
    global unigram_counts, bigram_counts, trigram_counts, total_tokens, doc_count

    if not os.path.exists(PICKLE_PATH):
        return False

    with open(PICKLE_PATH, 'rb') as f:
        state = pickle.load(f)

    documents = state.get('documents', {})
    doc_term_freq = state.get('doc_term_freq', {})
    df = state.get('df', defaultdict(int))
    doc_tf_idf = state.get('doc_tf_idf', {})
    unigram_counts = state.get('unigram_counts', Counter())
    bigram_counts = state.get('bigram_counts', Counter())
    trigram_counts = state.get('trigram_counts', Counter())
    total_tokens = state.get('total_tokens', 0)
    doc_count = state.get('doc_count', 0)
    return True


def initialize_corpus():
    global doc_count, documents, doc_term_freq

    loaded = load_state_from_pickle()
    if loaded:
        existing_files = set(documents.keys())
        all_files = {
            fn for fn in os.listdir(app.config['UPLOAD_FOLDER'])
            if fn.lower().endswith('.txt')
        }
        new_files = all_files - existing_files
        if new_files:
            for filename in new_files:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                tokens = preprocess_text(text)

                documents[filename] = tokens
                doc_term_freq[filename] = Counter(tokens)
                doc_count += 1
                build_ngrams(tokens)

            recompute_df_and_tfidf()
            visualisasi()
            save_state_to_pickle()

    else:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.lower().endswith('.txt'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                tokens = preprocess_text(text)

                documents[filename] = tokens
                doc_term_freq[filename] = Counter(tokens)
                doc_count += 1
                build_ngrams(tokens)

        if doc_count > 0:
            recompute_df_and_tfidf()
            visualisasi()
        save_state_to_pickle()


initialize_corpus()


@app.route('/', methods=['GET', 'POST'])
def index():
    global doc_count

    if 'upload' in request.form:
        files = request.files.getlist('documents')
        added = False
        for file in files:
            if file and file.filename.lower().endswith('.txt'):
                filename = secure_filename(file.filename)
                if filename in documents:
                    continue
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                tokens = preprocess_text(text)

                documents[filename] = tokens
                doc_term_freq[filename] = Counter(tokens)
                doc_count += 1
                build_ngrams(tokens)
                added = True

        if added:
            recompute_df_and_tfidf()
            visualisasi()
            save_state_to_pickle()

    stats = {
        'doc_count': doc_count,
        'num_unigram': sum(unigram_counts.values()),
        'num_bigram': sum(bigram_counts.values()),
        'num_trigram': sum(trigram_counts.values()),
        'total_tokens': total_tokens
    }

    has_visuals = (doc_count > 0)
    return render_template('index.html', stats=stats, has_visuals=has_visuals)


@app.route('/reset', methods=['POST'])
def reset():
    global documents, doc_term_freq, df, doc_tf_idf
    global unigram_counts, bigram_counts, trigram_counts, total_tokens, doc_count

    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.lower().endswith('.txt'):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    if os.path.exists(PICKLE_PATH):
        os.remove(PICKLE_PATH)

    documents.clear()
    doc_term_freq.clear()
    df.clear()
    doc_tf_idf.clear()
    unigram_counts.clear()
    bigram_counts.clear()
    trigram_counts.clear()
    total_tokens = 0
    doc_count = 0

    wc_path = os.path.join('static', 'wordcloud.png')
    top_path = os.path.join('static', 'top_tokens.png')
    if os.path.exists(wc_path):
        os.remove(wc_path)
    if os.path.exists(top_path):
        os.remove(top_path)

    return redirect(url_for('index'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Menerima JSON { "query": "<teks user>" }
    Mengembalikan JSON:
    {
      "suggestions": [ { "word": "...", "prob": 0.1234 }, ... ],
      "docs":        [ { "filename": "...", "score": 0.5678 }, ... ]
    }
    """
    data = request.get_json() or {}
    query_text = data.get('query', '').strip()
    tokens = preprocess_text(query_text)

    suggestions = []
    if len(tokens) == 1:
        prev = tokens[0]
        candidates = {
            w2: count
            for (w1, w2), count in bigram_counts.items()
            if w1 == prev
        }
        for w2, count in candidates.items():
            prob = count / unigram_counts[prev] if unigram_counts[prev] > 0 else 0.0
            suggestions.append((w2, prob))

    elif len(tokens) == 2:
        prev_bigram = tuple(tokens)
        candidates_tri = {
            w3: count
            for (w1, w2, w3), count in trigram_counts.items()
            if (w1, w2) == prev_bigram
        }
        if candidates_tri and prev_bigram in bigram_counts:
            for w3, count in candidates_tri.items():
                prob = count / bigram_counts[prev_bigram]
                suggestions.append((w3, prob))
        else:
            prev = tokens[1]
            candidates = {
                w2: count
                for (w1, w2), count in bigram_counts.items()
                if w1 == prev
            }
            for w2, count in candidates.items():
                prob = count / unigram_counts[prev] if unigram_counts[prev] > 0 else 0.0
                suggestions.append((w2, prob))

    suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)[:10]
    suggestions_json = [{ "word": w, "prob": p } for (w, p) in suggestions]

    docs_list = []
    if query_text and documents:
        query_tf = Counter(tokens)
        N = len(documents)
        query_vec = {}
        for term, tf in query_tf.items():
            idf = math.log((N) / df[term]) if df.get(term, 0) > 0 else 0.0
            query_vec[term] = tf * idf

        for fname, doc_vec in doc_tf_idf.items():
            sim = cosine_similarity(doc_vec, query_vec)
            if sim > 0:
                docs_list.append((fname, sim))

    docs_list = sorted(docs_list, key=lambda x: x[1], reverse=True)[:10]
    docs_json = [{ "filename": fn, "score": sc } for (fn, sc) in docs_list]

    return jsonify({
        "suggestions": suggestions_json,
        "docs": docs_json
    })


if __name__ == '__main__':
    app.run(debug=True)
