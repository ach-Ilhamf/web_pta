import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Inisialisasi NLTK (jalankan sekali)
nltk.download("punkt")
nltk.download("stopwords")

# Fungsi untuk normalisasi teks
def normalize_text(text):
    if isinstance (text, str):
        # Tokenisasi
        tokens = word_tokenize(text)

        # Normalisasi dengan menghapus tanda baca dan angka
        tokens = [re.sub(r'[.,():-]', '', token) for token in tokens]
        tokens = [re.sub(r'\d+', '', token) for token in tokens]

        # Menghapus stop words
        stop_words = set(stopwords.words("indonesian"))
        custom_stop_words = {"aas", "aatau", "ab", "atau", "yang", "dan", "ini", "itu", "tentu", "pasti", "tidak", "mungkin", "bisa", "zk", "zm", "zn", "zpt"}
        stop_words.update(custom_stop_words)
        tokens = [word for word in tokens if word not in stop_words]

        return " ".join(tokens)
    else:
        return ''  # Kembalikan string kosong jika data tidak valid

# Fungsi untuk stemming
def stem_text(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

# Fungsi untuk pemodelan topik menggunakan LDA
def lda_topic_modeling(text_data, num_topics=5, judul_abstrak=None):
    # Menghapus nilai-nilai NaN dari data sebelum pemodelan
    text_data = text_data.dropna()

    # Mengubah teks menjadi list token-token
    documents = text_data.apply(lambda x: x.split() if isinstance(x, str) else [])

    # Membuat representasi teks dalam bentuk "bag of words"
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Membuat dan melatih model LDA
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Menghitung proporsi topik dalam dokumen
    proporsi_topik = [lda_model.get_document_topics(doc) for doc in corpus]

    # Membuat DataFrame untuk menyimpan proporsi topik dalam dokumen
    proporsi_topik_dalam_dokumen = pd.DataFrame(columns=["Dokumen"] + [f"Topik_{i+1}" for i in range(lda_model.num_topics)])

    for i, doc_topic_proposals in enumerate(proporsi_topik):
        row_data = {"Dokumen": judul_abstrak[i] if judul_abstrak is not None and len(judul_abstrak) == len(proporsi_topik) else f"Dokumen {i + 1}"}
        for topic, prop in doc_topic_proposals:
            row_data[f"Topik_{topic + 1}"] = prop
        proporsi_topik_dalam_dokumen = pd.concat([proporsi_topik_dalam_dokumen, pd.DataFrame([row_data])], ignore_index=True)

    # Imputasi nilai-nilai NaN dengan 0
    proporsi_topik_dalam_dokumen = proporsi_topik_dalam_dokumen.fillna(0)

    # Menghitung Silhouette Score
    proporsi_kata_lda_matrix = proporsi_topik_dalam_dokumen.drop(columns=["Dokumen"]).to_numpy()
    kmeans_lda = KMeans(n_clusters=2)  # Ganti jumlah cluster sesuai kebutuhan
    kmeans_lda.fit(proporsi_kata_lda_matrix)
    lda_silhouette = silhouette_score(proporsi_kata_lda_matrix, kmeans_lda.labels_)

    # Menambahkan kolom "Kelas Cluster" yang berisi nomor cluster
    proporsi_topik_dalam_dokumen["Kelas Cluster"] = kmeans_lda.labels_

    return lda_model, proporsi_topik_dalam_dokumen, lda_silhouette

# Fungsi untuk pemodelan topik menggunakan TF-IDF
def tfidf_topic_modeling(text_data):
    # Membersihkan data dari nilai NaN
    text_data = text_data.dropna()

    # Membuat objek TfidfVectorizer dengan parameter tambahan
    vectorizer = TfidfVectorizer(use_idf=False, norm=None, binary=False)

    # Menghitung TF-IDF dari data teks
    tfidf_matrix = vectorizer.fit_transform(text_data)

    # Membuat DataFrame dari matriks TF-IDF dengan nama kolom yang sesuai
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    return tfidf_df

# Layout aplikasi
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.title("Pengaturan")
selected_tab = st.sidebar.radio("Pilih Tab:", ["Tab 1: Data", "Tab 2: Pemodelan Topik"])
uploaded_file = st.sidebar.file_uploader("Unggah File CSV", type=["csv"])

# Data placeholders
data = pd.DataFrame({'Abstrak': []})
lda_model = None
proporsi_kata_lda = None
lda_silhouette = None

# Memuat data dari file CSV yang diunggah
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

# Tab 1: Menampilkan Data
if selected_tab == "Tab 1: Data":
    st.title("Data")
    st.write(data)

# Tab 2: Pemodelan Topik
elif selected_tab == "Tab 2: Pemodelan Topik":
    st.title("Pemodelan Topik")
            
    st.subheader("Pilih Metode Pemodelan Topik:")
    topic_method = st.radio("Metode Pemodelan Topik:", ["LDA", "TF-IDF"])
        
    if topic_method == "LDA":
        data["normalisasi_abstrak"] = data["Abstrak"].apply(lambda x: normalize_text(x) if isinstance(x, str) else '')
        text_data = data["normalisasi_abstrak"]
        num_topics_lda = st.number_input("Jumlah Topik untuk LDA:", min_value=1, max_value=10, value=3, step=1)
        lda_model, proporsi_kata_lda, lda_silhouette = lda_topic_modeling(text_data, num_topics_lda, data["Judul"])
        st.write("Model LDA telah dibuat.")
        
        # Menampilkan matriks TF-IDF
        st.subheader("Proporsi topik dalam dokumen")
        st.write(proporsi_kata_lda)


        # Create tables for Cluster 0 and Cluster 1
        st.subheader("Tabel Cluster 0")
        cluster_0_df = proporsi_kata_lda[proporsi_kata_lda["Kelas Cluster"] == 0]
        st.write(cluster_0_df[["Dokumen"]])

        # Menghitung jumlah data dalam Cluster 0 dan 1 untuk metode LDA
        jumlah_data_cluster_0_lda = cluster_0_df.shape[0]
        st.subheader("Jumlah Data dalam Cluster 0 (Metode LDA):")
        st.write(jumlah_data_cluster_0_lda)

        st.subheader("Tabel Cluster 1")
        cluster_1_df = proporsi_kata_lda[proporsi_kata_lda["Kelas Cluster"] == 1]
        st.write(cluster_1_df[["Dokumen"]])

        jumlah_data_cluster_1_lda = cluster_1_df.shape[0]
        st.subheader("Jumlah Data dalam Cluster 1 (Metode LDA):")
        st.write(jumlah_data_cluster_1_lda)

        # Menampilkan Silhouette Score
        st.subheader("Silhouette Score untuk LDA Clustering:")
        st.write(lda_silhouette)


    elif topic_method == "TF-IDF":
        data["normalisasi_abstrak"] = data["Abstrak"].apply(lambda x: normalize_text(x) if isinstance(x, str) else '')
        data["stemmed_abstrak"] = data["normalisasi_abstrak"].apply(stem_text)
        text_data = data["stemmed_abstrak"]

        tfidf_df = tfidf_topic_modeling(text_data)
        st.write("Matriks TF-IDF telah dibuat.")
        
        # Menampilkan matriks TF-IDF
        st.subheader("Matriks TF-IDF")
        st.write(tfidf_df)

        # Perform K-Means clustering on TF-IDF data
        kmeans_tfidf = KMeans(n_clusters=2)  # Sesuaikan jumlah kelompok sesuai kebutuhan
        kmeans_tfidf.fit(tfidf_df)
        data["Kelas Cluster TF-IDF"] = kmeans_tfidf.labels_

        # Calculate the Silhouette Score
        silhouette_avg = silhouette_score(tfidf_df, kmeans_tfidf.labels_)


        # Display Cluster 0 and Cluster 1 in separate tables
        st.subheader("Tabel Cluster 0")
        cluster_0_df = data[data["Kelas Cluster TF-IDF"] == 0]
        st.write(cluster_0_df[["Judul"]])

        # Menghitung jumlah data dalam Cluster 0 dan 1 untuk metode TF-IDF
        jumlah_data_cluster_0_tfidf = cluster_0_df.shape[0]
        st.subheader("Jumlah Data dalam Cluster 0 (Metode TF-IDF):")
        st.write(jumlah_data_cluster_0_tfidf)

        st.subheader("Tabel Cluster 1")
        cluster_1_df = data[data["Kelas Cluster TF-IDF"] == 1]
        st.write(cluster_1_df[["Judul"]])

        jumlah_data_cluster_1_tfidf = cluster_1_df.shape[0]
        st.subheader("Jumlah Data dalam Cluster 1 (Metode TF-IDF):")
        st.write(jumlah_data_cluster_1_tfidf)
        
        # Display the Silhouette Score
        st.subheader("Skor Silhouette untuk Pengelompokan TF-IDF:")
        st.write(silhouette_avg)
