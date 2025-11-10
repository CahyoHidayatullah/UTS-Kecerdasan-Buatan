# UTS-Kecerdasan-Buatan
# 🚨 Deteksi Ujaran Kebencian Menggunakan Machine Learning

Proyek ini bertujuan untuk mendeteksi **ujaran kebencian (hate speech)** dalam teks menggunakan algoritma **Logistic Regression** dan **Naive Bayes** dengan representasi **TF-IDF**.  
Model ini dapat digunakan untuk membantu moderasi konten di media sosial atau forum publik.

---

## 📚 Daftar Isi
- [Latar Belakang](#-latar-belakang)
- [Tujuan](#-tujuan)
- [Dataset](#-dataset)
- [Metodologi](#-metodologi)
- [Preprocessing Data](#-preprocessing-data)
- [Pemodelan](#-pemodelan)
- [Evaluasi](#-evaluasi)
- [Hasil](#-hasil)
- [Kesimpulan](#-kesimpulan)
- [Cara Menjalankan](#-cara-menjalankan)
- [Struktur Proyek](#-struktur-proyek)


---

## 🎯 Latar Belakang

Ujaran kebencian merupakan salah satu bentuk komunikasi negatif yang dapat memicu konflik sosial dan menyebarkan diskriminasi di masyarakat.  
Deteksi manual terhadap ujaran kebencian tidak efisien karena jumlah data di media sosial sangat besar. Oleh karena itu, **teknik Kecerdasan Buatan (AI)** digunakan untuk mengotomatisasi proses ini.

Proyek ini membangun sistem klasifikasi teks sederhana untuk mendeteksi apakah suatu kalimat mengandung ujaran kebencian (*hate speech*) atau tidak (*non-hate*).

---

## 🧩 Tujuan

1. Mengembangkan model pembelajaran mesin untuk klasifikasi ujaran kebencian.
2. Melakukan preprocessing teks untuk menghasilkan fitur yang bersih dan bermakna.
3. Membandingkan performa model **Logistic Regression** dan **Multinomial Naive Bayes**.
4. Mengukur performa model menggunakan metrik seperti Accuracy, Precision, Recall, dan F1-Score.

---

## 📊 Dataset

Dataset yang digunakan merupakan data publik (contoh: [Davidson et al., 2017 Hate Speech Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language)) atau dataset lokal Bahasa Indonesia dengan format:

| text | label |
|------|--------|
| "Saya benci orang itu" | 1 |
| "Selamat pagi semuanya" | 0 |

Keterangan:
- `label = 1` → Hate Speech  
- `label = 0` → Non Hate Speech  

Jika kamu memiliki dataset lokal (`hate_speech.csv`), letakkan di folder `data/`.

---

## ⚙️ Metodologi

Tahapan penelitian meliputi:

1. **Pengumpulan Data**  
   Menggunakan dataset publik atau hasil crawling media sosial.

2. **Preprocessing**  
   - Case folding (huruf kecil)  
   - Hapus URL, angka, tanda baca  
   - Stopword removal (bahasa Indonesia/Inggris)  
   - Stemming (menggunakan `Sastrawi`)

3. **Ekstraksi Fitur**  
   Menggunakan **TF-IDF (Term Frequency – Inverse Document Frequency)** untuk merepresentasikan teks menjadi vektor numerik.

4. **Pemodelan**  
   - Logistic Regression  
   - Multinomial Naive Bayes  

5. **Evaluasi**  
   Menggunakan metrik:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix

---

## Soure Code

# 1️⃣ Import Library

         import pandas as pd
         import numpy as np
         import matplotlib.pyplot as plt
         from wordcloud import WordCloud
         from sklearn.model_selection import train_test_split
         from sklearn.feature_extraction.text import TfidfVectorizer
         from sklearn.naive_bayes import MultinomialNB
         from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
         import re
         import nltk
         from nltk.corpus import stopwords
         nltk.download('stopwords')

# 2️⃣ Load Dataset (Contoh dataset simulasi)

         data = {
             'text': [
                 'Kamu bodoh sekali!',
                 'Aku suka produk ini',
                 'Dasar goblok dan tolol',
                 'Saya sangat senang dengan layanan ini',
                 'Kamu hina sekali',
                 'Pelayanan sangat buruk',
                 'Hebat sekali kerja tim ini'
             ],
             'label': [1, 0, 1, 0, 1, 1, 0]
         }
         df = pd.DataFrame(data)
         print('Jumlah data:', len(df))
         df.head()

# 3️⃣ Preprocessing

         def clean_text(text):
             text = text.lower()
             text = re.sub(r'[^a-zA-Z\s]', '', text)
             tokens = text.split()
             stop_words = set(stopwords.words('indonesian'))
             tokens = [word for word in tokens if word not in stop_words]
             return ' '.join(tokens)
         
         df['clean_text'] = df['text'].apply(clean_text)
         df[['text', 'clean_text']]

# 4️⃣ Representasi Fitur (TF-IDF)

         vectorizer = TfidfVectorizer()
         X = vectorizer.fit_transform(df['clean_text'])
         y = df['label']

# 5️⃣ Split Data

         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Pembangunan Model (Naive Bayes)

         model = MultinomialNB()
         model.fit(X_train, y_train)

# 7️⃣ Evaluasi Model
         y_pred = model.predict(X_test)
               
               acc = accuracy_score(y_test, y_pred)
               prec = precision_score(y_test, y_pred)
               rec = recall_score(y_test, y_pred)
               f1 = f1_score(y_test, y_pred)
               
               print('\n=== Evaluasi Model ===')
               print(f'Akurasi   : {acc:.2f}')
               print(f'Presisi   : {prec:.2f}')
               print(f'Recall    : {rec:.2f}')
               print(f'F1-Score  : {f1:.2f}')
               
               cm = confusion_matrix(y_test, y_pred)
               disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Tidak Hate', 'Hate'])
               disp.plot(cmap=plt.cm.Blues)
               plt.title('Confusion Matrix Model Naive Bayes')
               plt.show()

# 8️⃣ Visualisasi Distribusi Label

               plt.figure(figsize=(5,4))
               df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
               plt.xticks([0,1], ['Tidak Hate', 'Hate'])
               plt.title('Distribusi Label Dataset')
               plt.xlabel('Kategori')
               plt.ylabel('Jumlah')
               plt.show()

# 9️⃣ Word Cloud Ujaran Kebencian

               hate_texts = ' '.join(df[df['label']==1]['clean_text'])
               wordcloud = WordCloud(width=800, height=400, background_color='white').generate(hate_texts)
               
               plt.figure(figsize=(10,5))
               plt.imshow(wordcloud, interpolation='bilinear')
               plt.axis('off')
               plt.title('Word Cloud - Ujaran Kebencian')
               plt.show()
