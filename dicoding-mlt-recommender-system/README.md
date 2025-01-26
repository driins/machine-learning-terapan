# Laporan Proyek Machine Learning - Sistem Rekomendasi Buku

---

  

Nama: Indri Windriasari

  

email: indriwindriasari2511@gmail.com

  
  
  

## Project Overview

  

  

Sistem rekomendasi telah menjadi salah satu solusi utama untuk membantu pengguna menemukan item yang relevan di era digital. Dalam dunia literasi, lonjakan jumlah buku yang diterbitkan setiap tahunnya menciptakan tantangan besar bagi pengguna untuk menemukan buku yang sesuai dengan preferensi mereka. Jumlah koleksi buku dalam perpustakaan digital membuat pengguna kewalahan untuk memilih buku yang relevan [[1]](https://doi.org/10.1007/s10639-021-10643-8).

  

Sistem rekomendasi menjawab tantangan ini dengan memberikan rekomendasi yang dipersonalisasi berdasarkan data dan preferensi pengguna. Salah satu pendekatan yang sering digunakan dalam pengembangan sistem rekomendasi adalah _content-based filtering_ yang memanfaatkan atribut buku seperti genre, penulis, atau deskripsi, dan _collaborative filtering_ yang memanfaatkan pola interaksi pengguna seperti penilaian (_rating_) atau riwayat pembelian [[2]](https://pubs.ascee.org/index.php/iota/article/view/693), [[3]](https://www.semanticscholar.org/paper/A-book-recommendation-system-based-on-named-Sariki-Kumar/8649f772ce38b6f2de21cbc32d6f1479a142f833?utm_source=consensus). Studi juga menunjukkan bahwa sistem rekomendasi berbasis data besar dengan algoritma seperti TF-IDF dan cosine similarity dapat meningkatkan hasil rekomendasi [[4]](https://www.semanticscholar.org/paper/IMPLEMENTATION-OF-ONLINE-BOOK-RECOMMENDATION-SYSTEM-Kamath-Kumar/0200795bf38e459575ae1f6b25cf6bce3b2bd494?utm_source=consensus).

  

Dengan demikian, pengembangan sistem rekomendasi buku berbasis machine learning dapat menjadi langkah untuk membantu pengguna menemukan buku yang relevan secara secara lebih efektif.

  
  
  

## Business Understanding

  

### Pernyataan Masalah

  

Bagaimana cara mempersiapkan dan mengolah data mentah yang terdiri dari informasi buku, pengguna, dan rating agar dapat digunakan dalam pemodelan machine learning? Serta, bagaimana membangun model rekomendasi yang dapat secara efektif merekomendasikan buku sesuai dengan preferensi setiap pengguna?

  

### Tujuan (Goals)

  

Tujuan utama dari proyek ini adalah untuk **menghasilkan model sistem rekomendasi yang mampu membantu pengguna menemukan buku yang sesuai dengan minat dan preferensi pengguna**.

  

### Solution Approach

  

Untuk meraih tujuan dalam mengembangkan sistem rekomendasi buku tersebut, beberapa langkah utama yang dilakukan :

- **Mempersiapkan data berkualitas**, termasuk melakukan pembersihan data dan transformasi, sebagai dasar yang kuat untuk pemodelan.

- **Mengembangkan model machine learning** yang mampu memahami pola dalam data pengguna dan buku, serta menghasilkan rekomendasi yang personal dan akurat berdasarkan preferensi pengguna.

- **Melakukan evaluasi** untuk memastikan bahwa sistem rekomendasi memberikan hasil yang relevan dan sesuai dengan kebutuhan pengguna.

  

Dua pendekatan utama yang diterapkan untuk pemodelan yaitu **Content-based Filtering** dan **Collaborative Filtering**.

  

#### **1). Content-based Filtering**

  

Pendekatan ini akan menggunakan algoritma untuk memberikan rekomendasi buku yang relevan berdasarkan analisis fitur konten buku. Dua hal penting yang akan digunakan dalam pendekatan ini:

  

**TF-IDF Vectorizer**

TF-IDF (Term Frequency - Inverse Document Frequency) digunakan untuk mengubah teks mentah menjadi representasi angka yang dapat digunakan oleh model machine learning. Dengan menggunakan TF-IDF, sistem dapat menghitung relevansi kata atau term dalam dokumen (buku) dan memberi bobot berdasarkan frekuensi kata tersebut muncul dalam dokumen.

Rumus perhitungan IDF adalah sebagai berikut:

$idf_i = \log\left(\frac{n}{df_i}\right)$

Di mana:

- $idf_i$ adalah skor IDF untuk kata $i$

- $df_i$ adalah jumlah dokumen yang mengandung kata $i$

- $n$ adalah jumlah total dokumen

Sedangkan TF-IDF dihitung dengan mengalikan skor frekuensi kata (TF) dengan skor IDF:

  

$w_{i,j} = tf_{i,j} \times idf_i$

  

Di mana:

- $w_{i,j}$ adalah skor TF-IDF untuk kata $i$ pada dokumen $j$

- $tf_{i,j}$ adalah frekuensi kata $i$ pada dokumen $j$

  
  

**Cosine Similarity**

Cosine Similarity digunakan untuk mengukur derajat kesamaan antara dua vektor, yaitu antara buku yang telah dibaca oleh pengguna dan buku lainnya dalam katalog. Skor cosine similarity dihitung sebagai:

$S_c(A,B) = \cos(\theta) = \frac{A \times B}{|A| |B|}$

  

Dengan $A_i$ dan $B_i$ merupakan komponen dari masing-masing vektor $A$ dan $B$.

  

#### **2). Collaborative Filtering**

  

Pendekatan ini akan menganalisis pola perilaku pengguna, seperti rating yang diberikan kepada buku, untuk memberikan rekomendasi berdasarkan kesamaan preferensi dengan pengguna lain. Dalam pendekatan ini, diperlukan langkah persiapan data tambahan, yaitu:

  

**Penyandian Fitur**: Fitur-fitur seperti User-ID dan ISBN diubah menjadi indeks integer untuk memudahkan pemodelan.

  

Meskipun Collaborative Filtering dapat memberikan rekomendasi yang lebih luas berdasarkan data pengguna lain, pendekatan ini memiliki kelemahan, yaitu tidak dapat memberikan rekomendasi untuk buku yang belum memiliki rating atau interaksi.

  
  
  

## Data Understanding

  

### Dataset

  

Dataset yang digunakan adalah *[Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)* yang tersedia di Kaggle Dataset. Dataset ini dikumpulkan oleh **Cai-Nicolas Ziegler** selama periode 4 minggu (Agustus/September 2004) dari komunitas Book-Crossing dengan izin dari **Ron Hornbaker**, CTO Humankind Systems yang terdiri dari tiga file csv dengan informasi sebagai berikut:

  

### Informasi Dataset

  

**Ukuran Dataset**:

- `Users.csv`: 278,858 entri.

- `Books.csv`: 271,379 entri.

- `Ratings.csv`: 1,149,780 entri.

  

**Deskripsi Variabel**:

  

`Users.csv`

- Berisi ID pengguna yang telah dianonimkan serta data demografi.

- Fitur Utama:

- `User-ID`: ID unik pengguna yang dianonimkan.

- `Location`: Lokasi geografis pengguna.

- `Age`: Usia pengguna (NaN ketika tidak tersedia).

  

`Books.csv`

  

- Berisi informasi tentang buku yang diidentifikasi dengan ISBN.

- Fitur Utama:

- `ISBN`: Identifier unik untuk setiap buku.

- `Book-Title`: Judul buku.

- `Book-Author`: Nama penulis buku (hanya penulis pertama yang dicantumkan jika ada lebih dari satu).

- `Year-Of-Publication`: Tahun penerbitan buku.

- `Publisher`: Penerbit buku.

- `Image-URL-S`, `Image-URL-M`, `Image-URL-L`: Tautan ke gambar sampul buku dalam ukuran kecil, sedang, dan besar.

  

`Ratings.csv`

- Berisi informasi rating buku yang diberikan oleh pengguna, baik secara eksplisit maupun implisit.

- Fitur Utama:

- `User-ID`: ID pengguna yang memberikan rating.

- `ISBN`: Identifier buku yang dinilai.

- `Book-Rating`: Nilai rating:

- **Eksplisit**: Skala 1–10 (nilai lebih tinggi menunjukkan apresiasi yang lebih tinggi).

- **Implisit**: Diwakili oleh 0 (misalnya, ketika pengguna berinteraksi dengan buku tetapi tidak memberikan rating).

  

**Kondisi Dataset**:

1. Missing Values

  

Dataset memiliki nilai yang hilang. Hal ini dikonfirmasi melalui fungsi `explore()` dan `df.isnull().sum()`.

![fungsi explore](https://images2.imgbox.com/bc/67/krmRUPtg_o.png)

  

Berikut merupakan perolehan missing value pada data:

- `users_df`memiliki missing value pada atribut `Age`.

![users_df](https://images2.imgbox.com/c4/bc/AgETxAFd_o.png)

Persentase data sebesar 38.7% menunjukkan hampir setengah dari keseluruhan data user ini missing sehingga dapat dipertimbangkan untuk mengisi missing value dengan nilai yang paling sering muncul.

- `books_df` memiliki missing value pada atribut`Book-Author`, `Publisher`, dan `Image-URL-L`.

![books_df](https://images2.imgbox.com/4d/76/i8hHGoXV_o.png)

Penanganan missing value pada bagian ini dapat dipertimbangkan dihapus dengan menggunakan fungsi `.dropna()`. Hal ini karena jumlah yang sangat kecil sehingga tidak akan memengaruhi hasil training secara signifikan.

- `ratings_df` tidak memiliki missing value.

![ratings_df](https://images2.imgbox.com/9e/60/3wL9fS08_o.png)

3. Duplicate Values

  

Tidak ditemukan data duplikat dalam dataset. Proses pengecekan dilakukan menggunakan `df.duplicated().sum()` menunjukkan hasil 0 untuk setiap dataframe.

  

5. Distribusi Data

  

![rating_0](https://images2.imgbox.com/66/9f/DAISurfm_o.png)

  

Tidak terdapat missing value pada `ratings_df`. Namun, berdasarkan hasil visualisasi, rating yang terbanyak pada dataset adalah rating 0. Hal ini menunjukkan bahwa sebagian besar data tidak memiliki rating eksplisit. Data dengan rating 0 dapat menyebabkan bias dalam analisis, sehingga perlu dipertimbangkan untuk dihapus pada tahap data preparation. Hal ini dilakukan untuk memastikan hasil analisis dan rekomendasi menjadi lebih akurat.

  
  

## Data Preparation

Tahapan ini merupakan proses krusial untuk memastikan bahwa data yang digunakan dalam pengembangan model machine learning bersih dan relevan. Berikut penjelasan setiap langkah yang dilakukan:

#### **1. Penggabungan Data**

  

Pada tahap ini, data dari berbagai sumber seperti **rating**, **book**, dan **user** digabungkan untuk mendapatkan pandangan menyeluruh. Proses ini fokus pada penghubungan data `ISBN` (book) dan `user_id` (user) untuk analisis lebih lanjut dengan memastikan bahwa data yang digunakan saling terhubung.
  

**Teknik yang Digunakan**:

  

- Mengambil **nilai unik** dari kolom `user_id` pada `ratings_df` dan `users_df`, lalu digabungkan menggunakan `concatenate()` untuk melihat cakupan data pengguna.

- Mengambil **nilai unik** dari kolom `ISBN` pada `ratings_df` dan `books_df`, kemudian digabungkan dengan `concatenate()` untuk melihat cakupan data buku.

  

#### **2. Mengatasi Missing Value**

**Missing value** adalah elemen data yang hilang atau tidak ada, yang dapat memengaruhi analisis dan kinerja model jika tidak ditangani dengan baik. Dengan menangani missing values, data menjadi lebih lengkap dan terhindar dari bias akibat data yang tidak lengkap.

  

**Teknik yang Digunakan**:

  

- **Mengidentifikasi missing values** menggunakan  `.isnull()` dan `explore()`.

- **Mengisi data hilang**: Menggunakan nilai statistik seperti **mean**, **median**, atau **modus**.
- **Menghapus data hilang**: Baris atau kolom yang memiliki sedikit missing values dihapus jika tidak signifikan terhadap analisis.

  

**Kolom `Age` di `users_df`**:

  

- Memiliki missing value sebesar **34.75%**. Missing value diisi dengan **nilai modus (mode)** menggunakan `.fillna()`. Pendekatan ini dipilih karena dapat mencerminkan pola umum data:

  

users_df['Age'] = users_df['Age'].fillna(users_df['Age'].mode()[0])

  

**Kolom `Book-Author`, `Publisher`, dan `Image-URL-L`**:

  

- Missing value dihapus menggunakan `.dropna()` karena jumlah missing value sangat kecil sehingga tidak memengaruhi analisis secara signifikan.

  

`books_df = books_df.dropna()`

  

**Penyesuaian Data Rating**

  

Kolom rating tidak memiliki missing value, tetapi data dengan rating 0 dianggap tidak relevan karena berpotensi menimbulkan bias. Rating 0 tidak digunakan, sehingga hanya data dengan rating 1 hingga 10 yang digunakan.

  

Pada hasil data understanding terdapat terdapat **716.109** data dengan rating 0. Setelah dilakukan penyesuaian, hanya rating 1–10 yang digunakan sebagai berikut:

![rating_tanpa_0](https://images2.imgbox.com/eb/08/C1ogbvxM_o.png)

  
  
  

#### **3. Menghapus Data Duplikat**

  

**Data Duplikat** merupakan data yang muncul lebih dari satu kali dan memiliki nilai identik pada setiap kolom. Data ini dapat menyebabkan bias karena distribusi data tidak akurat.

  

**Teknik yang Digunakan**:

  

- Data duplikat diidentifikasi dengan `.duplicated()`.

- Baris yang duplikat dihapus menggunakan `.drop_duplicates()`.

  

Pada dataset ini, **tidak ditemukan data duplikat**, sehingga tidak diperlukan langkah penanganan lebih lanjut.

  
  
  

#### **4. Merge Data**

  

Proses penggabungan (merge) dilakukan untuk mengintegrasikan data buku dan data rating yang telah difilter. Hal ini memastikan setiap entri pada dataset gabungan memiliki data yang relevan, seperti rating buku dan informasi buku terkait. Dataframe `books_ratings_merge` berisi informasi gabungan dari kedua dataframe. Isinya mencakup data rating buku dari hasil filter`ratings_df` dan informasi buku dari `books_df`, di mana setiap baris menunjukkan rating untuk buku tertentu yang memiliki ISBN yang sama.

#### **5. Content Based Filtering**

Sebelum membangun model content-based filtering, diperlukan proses untuk menyiapkan data sehingga bisa dimanfaatkan secara optimal. Pada tahap ini, fokus utamanya adalah mengekstraksi informasi dari data buku.

##### **Ekstraksi Fitur TF-IDF**

**TF-IDF (Term Frequency-Inverse Document Frequency)** dilakukan untuk mengukur pentingnya sebuah kata dalam sebuah dokumen relatif terhadap kumpulan dokumen lainnya

- Kolom `book_author` digunakan sebagai sumber fitur untuk representasi konten.
-   Menggunakan **TF-IDF Vectorizer**, data teks diubah menjadi representasi numerik yang mencerminkan bobot relevansi setiap kata dalam dokumen.
-   Hasil ekstraksi berupa matriks dengan dimensi **(10.000, 5.575)**, di mana 10.000 mewakili jumlah buku dan 5.575 adalah jumlah kata unik yang diolah.

![tfidf](https://images2.imgbox.com/eb/2a/nE7ifQ5y_o.png)

  #### **6. Collaborative Filtering**
  
Proses collaborative filtering memerlukan data interaksi pengguna dengan item (dalam hal ini buku). Sebelum masuk ke tahap pemodelan, beberapa langkah persiapan dilakukan untuk memastikan data siap diproses.
  
  ##### **Encode Data**
  
  -  Kolom **`user_id`** dan **`ISBN`** (ID buku) diubah menjadi indeks integer. Langkah ini diperlukan agar data dapat diolah oleh algoritma yang memanfaatkan indeks numerik.
  - Contoh hasil: `user_id` "276726" menjadi indeks 0, dan `ISBN` "052165615X" menjadi indeks 1.
  
  ##### **Split Data**
  - Proses ini penting untuk menilai seberapa baik model dapat menggeneralisasi terhadap data yang belum terlihat.
  - Dataset dibagi menjadi **80% data training** dan **20% data validasi**.
  
  ![split data](https://images2.imgbox.com/91/15/hjklDt61_o.png)

## Modelling & Result

  

Sistem rekomendasi dirancang untuk memberikan rekomendasi buku kepada pengguna menggunakan dua pendekatan: **Content-Based Filtering** dan **Collaborative Filtering**. Karena ukuran dataset yang besar (mencapai ratusan hingga jutaan entri), subset data digunakan untuk efisiensi pemrosesan, yaitu **10.000 baris data buku** dan **5.000 baris data rating**.

  

### **Pendekatan 1: Content-Based Filtering**

  

Content-Based Filtering adalah metode yang merekomendasikan item (buku) berdasarkan atribut atau karakteristik dari item itu sendiri. Dalam konteks ini, kesamaan antara buku telah dihitung berdasarkan atribut nama penulis. Model memanfaatkan matriks hasil **TF-IDF Vectorizer** (dijelaskan di bagian sebelumnya) dan menghitung kesamaan antar buku menggunakan **Cosine Similarity**.

#### **Proses Modeling**

##### **Cosine Similarity**
- Digunakan untuk menghitung skor kesamaan antar buku berdasarkan hasil matriks TF-IDF.
-   Menghasilkan matriks kesamaan berukuran **(10.000 x 10.000)**, di mana setiap sel merepresentasikan skor kesamaan antara dua buku.

![cos_sim](https://images2.imgbox.com/72/6c/IFdhG0fu_o.png)

  

**Top-N Recommendation Output**:

Fungsi `book_recommendations()` dikembangkan untuk menghasilkan daftar rekomendasi berdasarkan kesamaan penulis. Rekomendasi diberikan untuk buku tertentu berdasarkan k buku teratas yang paling mirip.

Sistem menghasilkan daftar rekomendasi dengan memilih buku-buku yang memiliki skor kesamaan tertinggi terhadap buku target sebagai berikut

#### Contoh Hasil

  

Buku Tertarget: "Devoted" penulis "Alice Borchardt"

  

![targeted_book](https://images2.imgbox.com/ae/03/1YyxOYSZ_o.png)

  

Hasil berupa 10 rekomendasi yang mirip dengan buku tertarget:

  

![content_based_hasil](https://images2.imgbox.com/5d/93/Ll9WYfq5_o.png)

  

#### **Kelebihan dan Kekurangan**:

| **Kelebihan**                                                                              | **Kekurangan**                                                                           |
|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| + Efektif untuk buku baru tanpa data interaksi pengguna (mengatasi *cold-start problem*).  | - Tidak mempertimbangkan preferensi pengguna; rekomendasi kurang personal.               |
| + Tidak memerlukan data pengguna; hanya menggunakan atribut buku.                          | - Bergantung pada atribut buku, sehingga terbatas pada kualitas data deskriptif.         |
  

---

  

### **Pendekatan 2: Collaborative Filtering**

  

Pendekatan ini merekomendasikan buku berdasarkan pola interaksi pengguna, menggunakan data rating sebagai basis. Hubungan antar pengguna dan buku dipelajari melalui embedding.

  

#### **Proses Modelling**

Pendekatan ini menggunakan embedding untuk mempelajari hubungan antara pengguna dan buku. Embedding adalah representasi berdimensi rendah dari entitas (seperti `user_id` dan `book_id`), yang memungkinkan model menangkap pola hubungan yang kompleks. 

  

#### Detail Model:

| **Parameter**     | **Deskripsi**                                                                       |
|--------------------|-------------------------------------------------------------------------------------|
| **Model**         | RecommenderNet (menggunakan embedding untuk pengguna dan buku).                     |
| **Input**         | - Jumlah pengguna: 1204.                                                            |
|                    | - Jumlah buku: 4565.                                                               |
|                    | - Embedding size: 50.                                                              |
| **Optimizer**     | Adam                                                                                |
| **Loss Function** | Binary Crossentropy                                                                 |
| **Epoch**         | 100                                                                                 |


  
  

#### Hasil Top-N Recommendation

Setelah pelatihan, sistem menghasilkan rekomendasi dengan memanfaatkan pola interaksi pengguna sebagai berikut:

  
  
  

![collab_hasil](https://images2.imgbox.com/9e/ec/kA0Cmnu7_o.png)

  

Dari hasil tersebut, bagian **Book with high ratings from user** menampilkan buku-buku yang telah diberi rating tinggi oleh pengguna terpilih (dalam contoh ini, pengguna dengan userId = 805).

Daftar ini membantu memverifikasi preferensi pengguna sebelum memberikan rekomendasi, sehingga pengguna dapat melihat bahwa sistem mengenali buku-buku favoritnya.


Selanjutnya, sistem membandingkan buku-buku tersebut dengan koleksi buku lainnya, kecuali buku yang sudah dibaca oleh pengguna. Berdasarkan nilai rekomendasi tertinggi, sistem menghasilkan daftar 10 buku yang paling relevan untuk direkomendasikan seperti pada gambar di atas.

  

#### Kelebihan dan Kekurangan

  

| **Kelebihan**                                                                    | **Kekurangan**                                                                         |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| + Mampu menangkap hubungan kompleks antara pengguna dan buku.                    | - Memerlukan waktu training lebih lama dibanding metode sederhana.                    |
| + Fleksibel untuk dataset besar dengan embedding berdimensi rendah.              | - Membutuhkan tuning hyperparameter seperti ukuran embedding dan jumlah epoch.        |
| + Menghasilkan rekomendasi yang relevan sesuai preferensi pengguna.              | - Membutuhkan sumber daya komputasi yang lebih besar pada dataset yang sangat besar.  |


---

  

## Evaluation

  

### **Metrik Evaluasi**

  

#### 1. Content-Based Filtering Evaluation Metrics

  

Evaluasi pada pendekatan berbasis kesamaan penulis menghasilkan metrik sebagai berikut:

![evaluasi_content_based](https://images2.imgbox.com/70/9d/4CH7dX3N_o.png)

Hasil tersebut diperoleh dengan 257 sample yang berhasil dievaluasi.

**Precision@K**

$Precision@K = \frac{\text{Jumlah Rekomendasi Relevan Dalam K Teratas}}{K}$

$Precision@10 = \frac{\text{Jumlah Buku Dengan Penulis Sama}}{10}$

$Precision@10 = 33.9$


-   Hasil precision = 33.9% menunjukkan dari 10 rekomendasi teratas, rata-rata hanya sekitar 3-4 buku yang relevan (dengan penulis sama).
-   Keterbatasan precision disebabkan oleh model yang hanya mempertimbangkan kesamaan penulis tanpa aspek lain seperti genre atau tema.

**Recall**

$Recall = \frac{\text{Jumlah Rekomendasi Relevan}}{\text{Total Buku Relevan yang Tersedia}}$

$Recall = \frac{\text{Jumlah Buku Penulis Sama yang Direkomendasikan}}{\text{Total Buku dari Penulis Tersebut - 1}}$

$Recall = 0.394$

-   Model berhasil menemukan hampir 40% buku relevan yang tersedia.
-   Recall lebih tinggi dibanding precision karena model memprioritaskan menemukan buku sebanyak mungkin dari penulis yang sama.

**F1-Score**

$F1\text{-}Score = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

$F1Score = 29.4$

-   F1-Score yang rendah mencerminkan ketidakseimbangan antara precision dan recall.
-   Trade-off terjadi karena model fokus pada kesamaan penulis, sehingga mengorbankan variasi dalam rekomendasi.


#### 2. Collaborative Filtering Evaluation Metrics

Model Collaborative Filtering dievaluasi menggunakan **Root Mean Squared Error (RMSE)**, yang memberikan gambaran tentang jarak rata-rata antara prediksi dan nilai aktual.

  

$RMSE = \sqrt{\frac{\sum_{i=1}^n (y_i - y_{pred})^2}{n}}$

Dimana:

  

$n$ = jumlah dataset

  

$y_i$ = nilai aktual

  

$y_pred$ = nilai prediksi

  

RMSE yang rendah mengindikasikan prediksi model yang akurat, karena menunjukkan jarak yang kecil antara nilai prediksi dan nilai aktual. 

Evaluasi visualisasi performa model dapat dilihat melalui grafik training dan validation error RMSE serta grafik training dan validation loss berikut:

  

![grafik_evaluasi](https://images2.imgbox.com/4b/d7/8H4eisO5_o.png)


Grafik tersebut menunjukkan dua metrik selama 100 epoch training:
- Training and Validation Error (RMSE)
- Training and Validation Loss

**Analisis Metrik:**

**Root Mean Square Error (RMSE)**

- Training RMSE  : ~0.05 (final)
- Validation RMSE : ~0.25 (final)

Interpretasi:
- Training RMSE menurun signifikan dari 0.30 ke 0.05
- Validation RMSE stabil di sekitar 0.25
- Gap besar antara training dan validation RMSE mengindikasikan overfitting

**Loss Function**
- Training Loss  : ~0.47 (final)
- Validation Loss : ~0.62 (final)

Interpretasi:
- Training loss menurun stabil
- Validation loss lebih tinggi dan gap cenderung melebar
- Konfirmasi adanya overfitting

**Identifikasi Masalah**
- Model terlalu baik mempelajari data training
- Kurang baik dalam generalisasi ke data baru

Hal tersebut ditunjukkan oleh gap antara training dan validation metrics.

Maka dapat disimpulkan, model menunjukkan kemampuan pembelajaran yang baik (RMSE training 0.05) namun mengalami overfitting. Meskipun demikian, validation RMSE ~0.25 masih dapat diterima untuk sistem rekomendasi, dengan akurasi prediksi sekitar 75% (1 - 0.25).

**Analisis, Justifikasi Performa, dan Komparasi**
-   Dataset buku memiliki distribusi penulis yang tidak merata.
-   Beberapa penulis hanya memiliki sedikit buku, sehingga membatasi variasi rekomendasi.
- **Faktor Model**:
	-   Pendekatan Content-Based Filtering terbatas pada fitur penulis, sehingga precision rendah.
	-   Collaborative Filtering lebih fleksibel karena mempelajari hubungan pengguna dan buku melalui embedding, menghasilkan prediksi dengan RMSE rendah.
 
## Kesimpulan

  

Dengan mengimplementasikan kedua pendekatan **Content-based Filtering** dan **Collaborative Filtering** sistem rekomendasi buku yang dibangun akan dapat memberikan rekomendasi yang lebih personal, relevan, dan adaptif terhadap perubahan preferensi pengguna. 
-   **Content-Based Filtering**: Cocok untuk skenario dengan data interaksi pengguna terbatas dan data tanpa interaksi antar pengguna.
-   **Collaborative Filtering**: Memberikan performa yang lebih baik dengan prediksi yang akurat (RMSE rendah) dan generalisasi yang baik, membuatnya ideal untuk digunakan sebagai model utama dalam sistem rekomendasi.

Kedua teknik ini saling melengkapi, sehingga secara keseluruhan, model dapat diintegrasikan untuk menciptakan rekomendasi yang lebih baik dengan menggabungkan pendekatan Content-Based dan Collaborative Filtering untuk mengatasi kekurangan masing-masing dan memberikan rekomendasi yang lebih akurat dan beragam.

  

## Referensi

  

  

[1] Sharma, S., Rana, V., & Malhotra, M. (2021). Automatic recommendation system based on hybrid filtering algorithm. _Education and Information Technologies_, 27, 1523 - 1538. [https://doi.org/10.1007/s10639-021-10643-8](https://doi.org/10.1007/s10639-021-10643-8).

  

  

[2] Lailatul Rosidah, Prita Dellia. (2024). Library Book Recommendation System Using Content-Based Filtering. Internet of Things and Artificial Intelligence Journal

  

  

[3] Sariki, Tulasi Prasad and Bharadwaja Kumar. (2018). “A book recommendation system based on named entities.”.

  

  

[4] Nithin M Kamath, Pratyush Kumar, T. Sreenivas. (2016). IMPLEMENTATION OF ONLINE BOOK RECOMMENDATION SYSTEM. International Journal of Modern Trends in Engineering and Research