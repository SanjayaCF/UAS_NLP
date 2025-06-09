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
import html

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
doc_count = 0

df = defaultdict(int)
doc_tf_idf = {}

#preproses
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = re.findall(r'\b[a-z]+\b', text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return tokens

#hitung ngrams
def build_ngrams(tokens):
    global total_tokens
    for i, w in enumerate(tokens):
        unigram_counts[w] += 1
        total_tokens += 1
        if i < len(tokens)-1:
            bigram_counts[(w, tokens[i+1])] += 1
        if i < len(tokens)-2:
            trigram_counts[(w, tokens[i+1], tokens[i+2])] += 1

#hitung tf-idf
def recompute_df_and_tfidf():
    df.clear()
    for toks in documents.values():
        for t in set(toks):
            df[t] += 1
    N = len(documents)
    doc_tf_idf.clear()
    for fn, term_counts in doc_term_freq.items():
        vec = {}
        for term, tf in term_counts.items():
            idf = math.log(N/df[term]) if df[term]>0 else 0.0
            vec[term] = tf * idf
        doc_tf_idf[fn] = vec

#hitung cosine similarity
def cosine_similarity(v1, v2):
    dot = sum(v1.get(t,0)*v2.get(t,0) for t in v1)
    n1 = math.sqrt(sum(x*x for x in v1.values()))
    n2 = math.sqrt(sum(x*x for x in v2.values()))
    return dot/(n1*n2) if n1 and n2 else 0.0

#visualsasi wordcloud dan top 10 token
def visualisasi():
    wc = WordCloud(width=800, height=400, background_color='white')\
           .generate_from_frequencies(unigram_counts)
    wc.to_file(os.path.join('static','wordcloud.png'))

    top10 = unigram_counts.most_common(10)
    words, freqs = zip(*top10) if top10 else ([],[])
    plt.figure(figsize=(8,5))
    plt.bar(words, freqs)
    plt.title('Top 10 Token')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join('static','top_tokens.png'))
    plt.close()


#simpan ke corpus_data.pkl
def save_state_to_pickle():
    state = {
        'documents': documents,
        'doc_term_freq': doc_term_freq,
        'unigram_counts': unigram_counts,
        'bigram_counts': bigram_counts,
        'trigram_counts': trigram_counts,
        'total_tokens': total_tokens,
        'doc_count': doc_count,
        'df': df,
        'doc_tf_idf': doc_tf_idf
    }
    with open(PICKLE_PATH,'wb') as f:
        pickle.dump(state,f)

#load dari corpus_data.pkl
def load_state_from_pickle():
    global documents, doc_term_freq
    global unigram_counts, bigram_counts, trigram_counts, total_tokens, doc_count
    global df, doc_tf_idf
    if not os.path.exists(PICKLE_PATH):
        return False
    with open(PICKLE_PATH,'rb') as f:
        state = pickle.load(f)
    documents      = state['documents']
    doc_term_freq  = state['doc_term_freq']
    unigram_counts = state['unigram_counts']
    bigram_counts  = state['bigram_counts']
    trigram_counts = state['trigram_counts']
    total_tokens   = state['total_tokens']
    doc_count      = state['doc_count']
    df             = state['df']
    doc_tf_idf     = state['doc_tf_idf']
    return True

#load semua dari pickle dan jika ada file baru akan diproses sendiri lagi
def initialize_corpus():
    global doc_count
    if load_state_from_pickle():
        existing = set(documents.keys())
        all_files = {f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.txt')}
        new_files = all_files - existing
        for fn in new_files:
            path = os.path.join(app.config['UPLOAD_FOLDER'],fn)
            text = open(path,encoding='utf-8').read()
            toks = preprocess_text(text)
            documents[fn] = toks
            doc_term_freq[fn] = Counter(toks)
            doc_count += 1
            build_ngrams(toks)
        if new_files:
            recompute_df_and_tfidf()
            visualisasi()
            save_state_to_pickle()
    else:
        for fn in os.listdir(app.config['UPLOAD_FOLDER']):
            if fn.endswith('.txt'):
                path = os.path.join(app.config['UPLOAD_FOLDER'],fn)
                text = open(path,encoding='utf-8').read()
                toks = preprocess_text(text)
                documents[fn] = toks
                doc_term_freq[fn] = Counter(toks)
                doc_count += 1
                build_ngrams(toks)
        if doc_count>0:
            recompute_df_and_tfidf()
            visualisasi()
        save_state_to_pickle()

#langsung inisialisasi
initialize_corpus()


@app.route('/', methods=['GET','POST'])
def index():
    global doc_count
    if 'upload' in request.form:
        added = False
        for f in request.files.getlist('documents'):
            if f and f.filename.lower().endswith('.txt') and f.filename not in documents:
                fn = secure_filename(f.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'],fn)
                f.save(path)
                text = open(path,encoding='utf-8').read()
                toks = preprocess_text(text)
                documents[fn] = toks
                doc_term_freq[fn] = Counter(toks)
                doc_count += 1
                build_ngrams(toks)
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
    return render_template('index.html', stats=stats, has_visuals=(doc_count>0))

@app.route('/reset', methods=['POST'])
def reset():
    global documents, doc_term_freq
    global unigram_counts, bigram_counts, trigram_counts, total_tokens, doc_count
    global df, doc_tf_idf
    #delete .txt
    for fn in os.listdir(app.config['UPLOAD_FOLDER']):
        if fn.endswith('.txt'):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'],fn))
    #delete pickle
    if os.path.exists(PICKLE_PATH):
        os.remove(PICKLE_PATH)
    #reset variabel
    documents.clear(); doc_term_freq.clear()
    unigram_counts.clear(); bigram_counts.clear(); trigram_counts.clear()
    total_tokens=0; doc_count=0
    df.clear(); doc_tf_idf.clear()
    #delete visual
    for img in ('wordcloud.png','top_tokens.png'):
        p=os.path.join('static',img)
        if os.path.exists(p): os.remove(p)
    return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json() or {}
    query_text = data.get('query', '').strip()
    tokens = preprocess_text(query_text)

    #laplace smoothing
    suggestions, V = [], len(unigram_counts)
    if len(tokens) >= 2:
        bg = (tokens[-2], tokens[-1]); c_bg = bigram_counts.get(bg, 0)
        for (w1,w2,w3),cnt in trigram_counts.items():
            if (w1,w2)==bg:
                suggestions.append((w3,(cnt+1)/(c_bg+V)))
        for w in unigram_counts:
            if (bg[0],bg[1],w) not in trigram_counts:
                suggestions.append((w,1/(c_bg+V)))
    elif len(tokens) == 1:
        prev = tokens[0]; c_prev = unigram_counts.get(prev,0)
        for (w1,w2),cnt in bigram_counts.items():
            if w1==prev:
                suggestions.append((w2,(cnt+1)/(c_prev+V)))
        for w in unigram_counts:
            if (prev,w) not in bigram_counts:
                suggestions.append((w,1/(c_prev+V)))
    suggestions = sorted(suggestions, key=lambda x:x[1], reverse=True)[:10]
    suggestions_json = [{"word":w,"prob":p} for w,p in suggestions]

    #tf-idf retrieval
    docs_list = []
    if query_text and documents:
        q_tf = Counter(tokens)
        N = len(documents)
        q_vec = {t:cnt*math.log(N/df[t]) if df[t]>0 else 0 for t,cnt in q_tf.items()}
        for fn,d_vec in doc_tf_idf.items():
            sim = cosine_similarity(d_vec, q_vec)
            if sim>0:
                docs_list.append((fn,sim))
    docs_list = sorted(docs_list, key=lambda x:x[1], reverse=True)[:10]

    #preview
    pattern_tokens = [re.compile(re.escape(t), re.IGNORECASE) for t in tokens]
    docs_json = []
    for fn, sc in docs_list:
        path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        try:
            text = open(path, 'r', encoding='utf-8').read()
        except:
            text = ''

        sentences = re.split(r'(?<=[\.\?\!])\s+', text)

        chosen = ''
        for sent in sentences:
            if all(p.search(sent) for p in pattern_tokens):
                chosen = sent.strip()
                break

        if not chosen:
            for sent in sentences:
                if any(p.search(sent) for p in pattern_tokens):
                    chosen = sent.strip()
                    break

        if not chosen:
            first_tok = tokens[0]
            idx = text.lower().find(first_tok.lower())
            if idx != -1:
                start = max(0, idx-50)
                end   = min(len(text), idx+len(first_tok)+50)
                chosen = text[start:end].strip()
            else:
                chosen = text[:100].strip()

        words = chosen.split()
        idx_word = next((i for i,w in enumerate(words)
                         if any(p.search(w) for p in pattern_tokens)), len(words)//2)
        window = 10
        st = max(0, idx_word-window)
        ed = min(len(words), idx_word+window+1)
        snippet_words = words[st:ed]
        snippet = ' '.join(snippet_words)
        if st>0: snippet = '... ' + snippet
        if ed<len(words): snippet += ' ...'

        esc = lambda t: html.escape(t, quote=False)
        snippet_esc = esc(snippet)
        for p in pattern_tokens:
            snippet_esc = p.sub(lambda m: f'<strong>{esc(m.group(0))}</strong>', snippet_esc)

        docs_json.append({
            "filename": fn,
            "score":    sc,
            "snippet":  snippet_esc
        })

    return jsonify({
        "suggestions": suggestions_json,
        "docs":        docs_json
    })




@app.route('/api/overview', methods=['GET'])
def api_overview():
    top_uni = unigram_counts.most_common(10)
    top_bi  = [(" ".join(k),v) for k,v in bigram_counts.most_common(10)]
    top_tri = [(" ".join(k),v) for k,v in trigram_counts.most_common(10)]
    top_df  = sorted(df.items(), key=lambda x:x[1], reverse=True)[:10]
    return jsonify({
      'top_unigram': [{'token':t,'count':c} for t,c in top_uni],
      'top_bigram':  [{'token':t,'count':c} for t,c in top_bi],
      'top_trigram': [{'token':t,'count':c} for t,c in top_tri],
      'top_df':      [{'term':t,'df':f}     for t,f in top_df]
    })

if __name__ == '__main__':
    app.run(debug=True)
