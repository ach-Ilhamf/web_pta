import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Inisialisasi NLTK (jalankan sekali)
nltk.download("punkt")
nltk.download("stopwords")

# Download data training dari GitHub
url = 'https://raw.githubusercontent.com/ach-Ilhamf/data_csv/main/data_berita%20(1).csv'
df_train = pd.read_csv(url)

# Initialize df_test to None
df_test = None

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df_train["Berita"])

# Similaritas Cosine
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Graf berbasis similaritas Cosine
G = nx.Graph()

for i in range(len(cosine_sim)):
    G.add_node(i)

for i in range(len(cosine_sim)):
    for j in range(len(cosine_sim)):
        similarity = cosine_sim[i][j]
        if similarity > 0.1 and i != j:
            G.add_edge(i, j, weight=similarity)

# Klasifikasi menggunakan Naive Bayes
X_train = tfidf_matrix
y_train = df_train["Topik"]

naive_bayes_model = make_pipeline(StandardScaler(with_mean=False), MultinomialNB())
naive_bayes_model.fit(X_train, y_train)

# Streamlit App
def main():
    # Check if session state is not initialized
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None

    # Initialize df_test if it's not uploaded yet
    if 'df_test' not in st.session_state:
        st.session_state.df_test = None

    st.title("Ringkasan Berita & Klasifikasi")

    # Pilihan Tab
    tabs = ["Ringkasan Berita", "Hasil Klasifikasi"]
    selected_tab = st.sidebar.radio("Pilih Tab:", tabs)

    # Tab 2: Ringkasan Berita
    if selected_tab == "Ringkasan Berita":
        st.header("Tab 1: Ringkasan Berita")
        uploaded_file = st.file_uploader("Upload Data Testing (CSV)", type="csv")

        if uploaded_file is not None:
            st.session_state.df_test = pd.read_csv(uploaded_file)
            st.dataframe(st.session_state.df_test)

            index_berita = st.selectbox("Pilih Index Berita", st.session_state.df_test.index)

            preprocessing_option = st.checkbox("Gunakan Preprocessing")

            method_options = ["Closeness Centrality", "Page Rank", "Eigen Vector"]
            method_option = st.selectbox("Pilih Metode Ringkasan", method_options)

            if st.button("Ringkas dan Klasifikasi"):
                # Define X_test based on the selected index
                berita = st.session_state.df_test["Berita"].iloc[index_berita]
                X_test = tfidf_vectorizer.transform([berita])

                # Save X_test in session state
                st.session_state.X_test = X_test

                ringkasan = ringkas_berita(X_test, index_berita, preprocessing_option, method_option)
                st.text("Ringkasan Berita:")
                st.text(ringkasan)

    # Tab 3: Hasil Klasifikasi
    elif selected_tab == "Hasil Klasifikasi":
        st.header("Tab 2: Hasil Klasifikasi")

        # Check if X_test is saved in session state
        if st.session_state.X_test is not None:
            X_test = st.session_state.X_test

            if st.button("Klasifikasi Berita"):
                hasil_klasifikasi = klasifikasi_berita(X_test)
                st.text("Hasil Klasifikasi:")
                st.text(hasil_klasifikasi)

def ringkas_berita(X_test, index_berita, preprocessing_option, method_option):
    berita = st.session_state.df_test["Berita"].iloc[index_berita]

    if preprocessing_option:
        berita = preprocessing(berita)

    kalimat = nltk.sent_tokenize(berita)

    if method_option == "Closeness Centrality":
        centrality_scores = nx.closeness_centrality(G)
    elif method_option == "Page Rank":
        centrality_scores = nx.pagerank(G)
    elif method_option == "Eigen Vector":
        centrality_scores = nx.eigenvector_centrality(G)

    top_sentences = get_top_sentences(centrality_scores, kalimat)

    return "\n".join(top_sentences)

def preprocessing(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s.]', '', text)
    text = text.lower()

    stop_words = set(stopwords.words('indonesian'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]

    preprocessing_text = ' '.join(filtered_words)

    return preprocessing_text

def get_top_sentences(centrality_scores, sentences, num_sentences=3):
    sorted_indices = sorted(centrality_scores, key=centrality_scores.get, reverse=True)

    # Ensure that sorted_indices is not empty
    if not sorted_indices or len(sentences) == 0:
        return []

    top_indices = sorted_indices[:num_sentences]
    top_indices = [i for i in top_indices if i < len(sentences)]  # Filter indices that are within the range

    # Ensure that top_indices is not empty before accessing elements from the sentences list
    top_sentences = [sentences[i] for i in top_indices]

    # Display all sentences if fewer than num_sentences are available
    if len(top_sentences) < num_sentences:
        top_sentences = sentences[:num_sentences]

    return top_sentences

def klasifikasi_berita(X_test):
    hasil_klasifikasi = naive_bayes_model.predict(X_test)
    return hasil_klasifikasi[0]

if __name__ == "__main__":
    main()
