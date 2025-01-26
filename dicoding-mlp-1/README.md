# Laporan Proyek Machine Learning - Mengatasi Tantangan Attrition Karyawan dengan Prediksi Berbasis Machine Learning

Nama : Indri Windriasari <br>
Email: indriwindriasari2511@gmail.com

## Domain Proyek

Pergantian karyawan (atau **attrition**) merupakan tantangan utama bagi banyak perusahaan di berbagai sektor industri. Di era persaingan talenta yang ketat, mempertahankan karyawan terbaik menjadi semakin penting karena tingginya tingkat pergantian karyawan bukan hanya berdampak pada stabilitas tim, tetapi juga memiliki konsekuensi operasional dan ekonomi yang signifikan bagi perusahaan. [[1]](https://doi.org/10.1117/12.2628107  "[1]"). Sebagai contoh, sebuah perusahaan teknologi yang kehilangan talenta-talenta terbaiknya tidak hanya harus mengeluarkan biaya untuk rekrutmen dan pelatihan ulang, tetapi juga berpotensi kehilangan momentum inovasi dan pangsa pasar. Menurut Gallup, biaya pergantian seorang karyawan berkisar antara 1,5 hingga 2 kali gaji tahunan mereka, yang menunjukkan beban finansial yang substansial bagi organisasi. Lebih dari sekadar biaya finansial, attrition juga dapat berdampak pada rusaknya moral tim, mengganggu kontinuitas proyek, dan menghilangkan pengetahuan serta keahlian strategis yang berharga. Oleh karena itu, mengembangkan model prediksi attrition menjadi penting bagi perusahaan sebagai langkah proaktif dalam meminimalisir dampak negatif yang mungkin timbul dari atrisi.

  

Pendekatan berbasis data dapat membantu mengidentifikasi pola-pola yang mengindikasikan kemungkinan seseorang akan meninggalkan perusahaan, dan membantu identifikasi awal dengan model prediksi attrition. Prediksi attrition menjadi pendekatan strategis untuk mitigasi risiko dalam suatu perusahaan. Metode klasifikasi machine learning menawarkan solusi dengan mengkategorikan karyawan ke dalam dua kelas utama: "Bertahan" dan "Keluar". Namun, kompleksitas permasalahan terletak pada ketidakseimbangan data, di mana proporsi karyawan yang keluar secara signifikan lebih rendah dibandingkan karyawan yang bertahan. Maka, Exploratory Data Analysis (EDA) [[2]](https://doi.org/10.46243/jst.2022.v7.i09.pp01-11  "[2]") dan teknik-teknik resampling [[3]](https://doi.org/10.33387/jiko.v4i1.2561  "[3]"), [[4]](https://doi.org/10.1186/s40537-020-00390-x  "[4]") menjadi pendekatan kunci dalam mengatasi tantangan ketidakseimbangan data. Dengan ini maka identifikasi pola dan faktor-faktor yang berkontribusi terhadap attrition dapat dilakukan dengan menghasilkan model yang akurat dan menghasilkan wawasan yang komprehensif terkait atrisi karyawan.

  

---

  

## Business Understanding

  

### Problem Statements

1.  **Faktor-faktor utama penyebab pergantian karyawan (attrition) masih belum teridentifikasi secara jelas.** Hal ini menyulitkan perusahaan dalam mengambil langkah-langkah preventif yang efektif.

2.  **Perusahaan belum memiliki sistem yang dapat memprediksi potensi attrition karyawan.** Akibatnya, perusahaan seringkali kehilangan talenta secara tiba-tiba tanpa persiapan yang memadai.

3.  **Data historis menunjukkan adanya ketidakseimbangan kelas yang signifikan antara karyawan yang bertahan dan yang keluar.** Hal ini dapat mempengaruhi performa model prediksi yang akan dibangun.

  

### Goals

1.  **Mengidentifikasi faktor-faktor utama yang berkontribusi terhadap attrition karyawan.**

2.  **Membangun model klasifikasi biner untuk memprediksi potensi attrition karyawan (klasifikasi "Bertahan" atau "Keluar")**

3.  **Mengatasi masalah ketidakseimbangan data untuk memastikan model dapat memberikan hasil yang valid dan tidak condong pada kelas mayoritas**

  

### Solution Statements

Untuk mencapai tujuan-tujuan di atas, solusi yang akan diimplementasikan adalah sebagai berikut:

- Melakukan **Exploratory Data Analysis (EDA)** untuk memahami karakteristik data, mengidentifikasi pola-pola yang relevan, dan menemukan korelasi antara berbagai faktor dengan attrition. EDA akan membantu memahami data secara lebih mendalam dan menginformasikan pemilihan fitur untuk model.

- Menggunakan algoritma machine learning seperti **XGBoost, Random Forest, dan Logistic Regression** untuk membangun model klasifikasi biner yang dapat memprediksi potensi attrition karyawan. Algoritma-algoritma ini dipilih karena efektivitasnya dalam menangani masalah klasifikasi.

- Mengatasi ketidakseimbangan data menggunakan metode **SMOTE** dilengkapi **Random Undersampling**. SMOTE menciptakan data sintetis untuk kelas minoritas, sementara Random Undersampling akan mengurangi data kelas mayoritas secara acak.

  

---

  

## Data Understanding

  

### Dataset

Dataset yang digunakan adalah *[IBM HR Analytics Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data)* yang tersedia di Kaggle. Dataset ini terdiri dari 1.470 entri data dengan 35 fitur terkait karyawan, termasuk demografi, riwayat pekerjaan, dan status attrition.

  

### Uraian Fitur Pada Dataset

| **No** | **Nama Atribut**         | **Deskripsi**                                                       |
|--------|--------------------------|---------------------------------------------------------------------|
| 1      | Age                      | Usia karyawan                                                      |
| 2      | Gender                   | Jenis kelamin karyawan                                             |
| 3      | BusinessTravel           | Frekuensi perjalanan dinas karyawan                                |
| 4      | DailyRate                | Gaji harian karyawan                                               |
| 5      | Department               | Departemen tempat karyawan bekerja                                 |
| 6      | DistanceFromHome         | Jarak rumah ke tempat kerja (dalam mil)                            |
| 7      | Education                | Tingkat pendidikan yang telah dicapai karyawan                    |
| 8      | EducationField           | Bidang studi karyawan                                              |
| 9      | EmployeeCount            | Jumlah total karyawan di organisasi                                |
| 10     | EmployeeNumber           | Nomor unik identitas karyawan                                      |
| 11     | EnvironmentSatisfaction  | Kepuasan karyawan terhadap lingkungan kerja                        |
| 12     | HourlyRate               | Tarif gaji per jam karyawan                                        |
| 13     | JobInvolvement           | Tingkat keterlibatan karyawan dalam pekerjaannya                  |
| 14     | JobLevel                 | Tingkatan posisi pekerjaan karyawan                                |
| 15     | JobRole                  | Peran atau jabatan karyawan dalam organisasi                      |
| 16     | JobSatisfaction          | Kepuasan karyawan terhadap pekerjaannya                           |
| 17     | MaritalStatus            | Status perkawinan karyawan                                        |
| 18     | MonthlyIncome            | Penghasilan bulanan karyawan                                      |
| 19     | MonthlyRate              | Tarif gaji bulanan karyawan                                       |
| 20     | NumCompaniesWorked       | Jumlah perusahaan yang pernah karyawan bekerja                    |
| 21     | Over18                   | Apakah karyawan berusia di atas 18 tahun                          |
| 22     | OverTime                 | Apakah karyawan bekerja lembur                                    |
| 23     | PercentSalaryHike        | Persentase kenaikan gaji karyawan                                 |
| 24     | PerformanceRating        | Penilaian kinerja karyawan                                        |
| 25     | RelationshipSatisfaction | Kepuasan karyawan terhadap hubungan interpersonal di kerja        |
| 26     | StandardHours            | Jam kerja standar karyawan                                        |
| 27     | StockOptionLevel         | Tingkat opsi saham karyawan                                       |
| 28     | TotalWorkingYears        | Total tahun karyawan telah bekerja                                |
| 29     | TrainingTimesLastYear    | Jumlah pelatihan yang diikuti karyawan tahun lalu                 |
| 30     | WorkLifeBalance          | Persepsi karyawan tentang keseimbangan kerja dan kehidupan        |
| 31     | YearsAtCompany           | Jumlah tahun karyawan bekerja di perusahaan                       |
| 32     | YearsInCurrentRole       | Jumlah tahun karyawan dalam peran saat ini                        |
| 33     | YearsSinceLastPromotion  | Jumlah tahun sejak promosi terakhir karyawan                      |
| 34     | YearsWithCurrManager     | Jumlah tahun karyawan dengan manajer saat ini                     |
| 35     | Attrition                | Apakah karyawan meninggalkan organisasi                           |

  

### Kondisi Dataset

| **Atribut**                  | **Tipe Data** | **Persentase Missing** | **Jumlah Unik** | **Nilai Unik Contoh**                                  |
|------------------------------|---------------|-------------------------|------------------|-------------------------------------------------------|
| Age                          | int64         | 0.0%                   | 43               | [41, 49, 37, 33, 27, ...]                             |
| YearsInCurrentRole           | int64         | 0.0%                   | 19               | [4, 7, 0, 2, 5, ...]                                  |
| YearsAtCompany               | int64         | 0.0%                   | 37               | [6, 10, 0, 8, 2, ...]                                 |
| WorkLifeBalance              | int64         | 0.0%                   | 4                | [1, 3, 2, 4]                                          |
| TrainingTimesLastYear        | int64         | 0.0%                   | 7                | [0, 3, 2, 5, ...]                                     |
| TotalWorkingYears            | int64         | 0.0%                   | 40               | [8, 10, 7, 6, ...]                                    |
| StockOptionLevel             | int64         | 0.0%                   | 4                | [0, 1, 3, 2]                                          |
| StandardHours                | int64         | 0.0%                   | 1                | [80]                                                 |
| RelationshipSatisfaction     | int64         | 0.0%                   | 4                | [1, 4, 2, 3]                                          |
| PerformanceRating            | int64         | 0.0%                   | 2                | [3, 4]                                               |
| PercentSalaryHike            | int64         | 0.0%                   | 15               | [11, 23, 15, ...]                                     |
| NumCompaniesWorked           | int64         | 0.0%                   | 10               | [8, 1, 6, 9, ...]                                     |
| MonthlyRate                  | int64         | 0.0%                   | 1427             | [19479, 24907, ...]                                   |
| MonthlyIncome                | int64         | 0.0%                   | 1349             | [5993, 5130, ...]                                     |
| YearsSinceLastPromotion      | int64         | 0.0%                   | 16               | [0, 1, 3, 2, ...]                                     |
| JobSatisfaction              | int64         | 0.0%                   | 4                | [4, 2, 3, 1]                                          |
| YearsWithCurrManager         | int64         | 0.0%                   | 18               | [5, 7, 0, 2, ...]                                     |
| JobLevel                     | int64         | 0.0%                   | 5                | [2, 1, 3, ...]                                        |
| DailyRate                    | int64         | 0.0%                   | 886              | [1102, 279, ...]                                      |
| DistanceFromHome             | int64         | 0.0%                   | 29               | [1, 8, 2, ...]                                        |
| Education                    | int64         | 0.0%                   | 5                | [2, 1, 4, ...]                                        |
| EmployeeNumber               | int64         | 0.0%                   | 1470             | [1, 2, 4, ...]                                        |
| EnvironmentSatisfaction      | int64         | 0.0%                   | 4                | [2, 3, 4, 1]                                          |
| EmployeeCount                | int64         | 0.0%                   | 1                | [1]                                                  |
| HourlyRate                   | int64         | 0.0%                   | 71               | [94, 61, 92, ...]                                     |
| JobInvolvement               | int64         | 0.0%                   | 4                | [3, 2, 4, 1]                                          |
| Over18                       | object        | 0.0%                   | 1                | [Y]                                                  |
| JobRole                      | object        | 0.0%                   | 9                | [Sales Executive, Research Scientist, ...]           |
| EducationField               | object        | 0.0%                   | 6                | [Life Sciences, Other, Medical, ...]                 |
| Gender                       | object        | 0.0%                   | 2                | [Female, Male]                                       |
| Department                   | object        | 0.0%                   | 3                | [Sales, Research & Development, Human Resources]     |
| BusinessTravel               | object        | 0.0%                   | 3                | [Travel_Rarely, Travel_Frequently, Non-Travel]       |
| Attrition                    | object        | 0.0%                   | 2                | [Yes, No]                                            |
| OverTime                     | object        | 0.0%                   | 2                | [Yes, No]                                            |
| MaritalStatus                | object        | 0.0%                   | 3                | [Single, Married, Divorced]                          |

- Keberadaan data Lengkap. Tidak ada nilai yang hilang (isnull().sum() == 0), sehingga tidak diperlukan langkah imputasi atau penanganan data kosong.
- Tidak ada duplikasi data (duplicated().sum() == 0), yang menegaskan keunikan setiap baris dalam dataset.
- Karakteristik variabel dataset:
   - Variabel Numerik: Sebagian besar atribut adalah numerik (misalnya, Age, YearsAtCompany, dan MonthlyIncome), yang memungkinkan penggunaan analisis statistik seperti korelasi dan visualisasi.
   - Variabel Kategorikal: Beberapa variabel kategorikal seperti Attrition, JobRole, Gender, dan BusinessTravel memegang peran kunci dalam prediksi dan segmentasi. - --
   - Variabel target (Attrition) menjadi fokus utama untuk analisis prediktif.
- Terdapat fitur yang tidak informatif. Variabel dengan nilai tetap, seperti di bawah inu tidak memberikan informasi tambahan karena memiliki hanya satu nilai unik. Oleh karena itu, disarankan untuk menghapusnya di tahapan data preparation.
   - EmployeeCount: memiliki nilai konstan
   - StandardHours: memiliki nilai konstan
   - Over18: memiliki nilai konstan
   - EmployeeNumber: hanya berisi ID unik karyawan yang tidak relevan untuk prediksi


**Distribusi Data**

![dis](https://images2.imgbox.com/f6/cc/kRCfOkYR_o.png)

  

Visualisasi distribusi data menunjukkan bahwa proporsi karyawan yang bertahan (sekitar 84%) jauh lebih besar daripada yang keluar (sekitar 16%). Ketidakseimbangan data ini perlu ditangani dengan teknik yang tepat agar model prediksi dapat bekerja secara optimal. Rasio ketidakseimbangan dihitung sebagai berikut:

  

$$ \text{Imbalance Ratio} = \frac{N_{\text{majority}}}{N_{\text{minority}}} $$

  

Di mana

  

Nmajority: Jumlah data kelas mayoritas

Nminority: Jumlah data kelas minoritas

  

Pada dataset ini, rasionya adalah sekitar 5,2 : 1.

  

### Exploratory Data Analysis (EDA)

Temuan Utama dari Proses EDA:

-  **Jumlah Karyawan yang Keluar Jauh Lebih Sedikit.**

  

Dari data, terlihat bahwa sebagian besar karyawan (84%) tetap bekerja di perusahaan. Hanya sebagian kecil (16%) yang keluar. Ini disebut ketidakseimbangan data, dan perlu diatasi nanti saat membuat model agar hasilnya akurat.

  

![distribusi atrisi](https://images2.imgbox.com/f6/cc/kRCfOkYR_o.png)

-  **Gaji Kecil Berhubungan dengan Keputusan Keluar.**

  

Karyawan yang gajinya di bawah $5000 cenderung lebih sering keluar. Hal ini logis erjadi, karena gaji yang kurang memadai bisa menjadi alasan kuat untuk mencari pekerjaan lain. Jadi, fitur "Gaji" ini penting untuk diperhatikan dalam model.

  

![Kepuasan Gaji dengan Atrisi](https://images2.imgbox.com/10/0a/26eJ7c7k_o.png)

-  **Dibandingkan karyawan wanita, persentase karyawan pria yang keluar lebih tinggi (63.3% berbanding 36.7%).**

  

![distribusi atrisi berdasarkan gender](https://images2.imgbox.com/75/ae/TWgoBUbd_o.png)

-  **Karyawan berumur 20–30 tahun paling banyak keluar.**

  

Hal ini mungkin karena mereka masih mencari pengalaman dan peluang yang lebih baik di awal karir.

  

![Distribusi usia](https://images2.imgbox.com/2a/8b/0xbQ3zY1_o.png)

-  **Beberapa informasi dalam data yang ternyata sama untuk semua karyawan**, seperti `EmployeeCount`, `StandardHours`, dan `Over18`. Karena semuanya sama, informasi ini **tidak membantu membedakan karyawan yang keluar dan yang tidak**, sehingga perlu dihapus sebelum dilakukan traing model agar tidak membingungkan model yang akan dibangun. `EmployeeNumber`sebagai nomor identitas karyawan juga dihapus karena hanya berupa nomor urut dan **tidak memberikan informasi karakteristik karyawan**.

  

---

  

## Data Preparation

1.   **Penghapusan Fitur yang Tidak Relevan.**
		Proses persiapan data dimulai dengan menghapus fitur yang tidak diperlukan. Ada empat kolom yang dihilangkan dari dataset yaitu `EmployeeCount`, `StandardHours`, `Over18` dan `EmployeeNumber` karena tidak memberikan informasi tambahan bagi model.

		-   `EmployeeCount`: memiliki nilai konstan
		-   `StandardHours`: memiliki nilai konstan
		-   `Over18`: memiliki nilai konstan
		-   `EmployeeNumber`: hanya berisi ID unik karyawan yang tidak relevan untuk prediksi

2. **Pemisahan Fitur Kategorikal dan Numerikal**
	Dataset dibagi menjadi dua kelompok fitur. Hal ini karena fitur numerik dan kategorikal memerlukan teknik preprocessing yang berbeda:
	-   Fitur kategorikal memerlukan encoding.
	-   Fitur numerik dapat distandarisasi atau dinormalisasi untuk menghindari bias fitur dengan skala besar.

	Berikut merupakan hasil pemisahan fitur dari dataset:
	
	-   **Fitur Numerikal**: Age, DailyRate, DistanceFromHome, Education, EnvironmentSatisfaction, HourlyRate, JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome, MonthlyRate, NumCompaniesWorked, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
	-   **Fitur Kategorikal**: Attrition, BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime

3. **Label Encoding**

   Algoritma machine learning tidak dapat bekerja langsung dengan data non-numerik. Encoding memastikan fitur kategorikal dapat diproses oleh model tanpa kehilangan informasi struktural. Langkah ini mengubah semua variabel kategorikal berbentuk string menjadi format numerik, membuatnya sesuai untuk algoritma machine learning sambil mempertahankan hubungan kategorikal dalam data. Proses ini dilakukan pada semua fitur kategorikal kecuali fitur target (Attrition).
	
	Sebagai contoh:
	Gender: Male = 0, Female = 1.

5. **Penanganan Ketidakseimbangan Data:**

	Dataset menunjukkan ketidakseimbangan dengan rasio 5.2:1 antara karyawan yang bertahan dan yang keluar. Ketidakseimbangan dapat membuat model cenderung memprediksi kelas mayoritas (`Bertahan`) sehingga mengurangi akurasi pada kelas minoritas (`Keluar`). Untuk mengatasi ini:
	1.  Dilakukan oversampling menggunakan SMOTE dengan sampling_strategy=0.85 pada kelas minoritas.
	2.  Dilanjutkan dengan undersampling menggunakan RandomUnderSampler untuk undersampling kelas mayoritas.
	3.  Hasil akhir menunjukkan distribusi yang lebih seimbang antara kedua kelas

	Pendekatan oversampling menggunakan SMOTE (Synthetic Minority Oversampling Technique) dilakukan untuk menambahkan data sintetis pada kelas minoritas:

$$ Xbaru=X+α⋅(Xtetangga−X) $$

   Di mana alpha adalah faktor skala acak.
   
   Pipeline ini diikuti oleh undersampling untuk memastikan keseimbangan data yang lebih baik.
   
   Hasil proses resampling:
   
   	> Distribusi kelas asli: Counter({0: 1233, 1: 237})
   
   	> Distribusi kelas setelah oversampling: Counter({0: 1233, 1: 1048})
   
   	> Distribusi kelas setelah undersampling: Counter({0: 1048, 1: 1048})


5. **Data Splitting**
	
 Data perlu dipisahkan untuk menguji kemampuan model generalisasi pada data baru. Rasio 80:20 dipilih untuk memastikan cukup data untuk pelatihan dan pengujian.

6. **Seleksi Fitur**

-  **Uji Chi-Squared (χ²):**

Menguji apakah ada hubungan signifikan antara fitur kategorikal dan target. . Nilai uji dihitung dengan rumus:

$$  \chi^2 = \sum  \frac{(O_i - E_i)^2}{E_i} $$

7. **Standarisasi dan Normalisasi**

-  **Normalisasi** menggunakan Min-Max Scaling diterapkan untuk fitur dengan rentang besar:

$$ X' = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}} $$

- Sementara fitur lainnya dinormalisasi menggunakan **Standarisasi (Z-Score):**

$$ Z = \frac{X - \mu}{\sigma} $$

  

Di mana:

-  \( O_i \): Frekuensi yang diamati

-  \( E_i \): Frekuensi yang diharapkan

-  **ANOVA F-Test:**

Mengukur variabilitas antara grup dibandingkan dengan variabilitas di dalam grup untuk fitur numerik. Nilai F dihitung sebagai:

  

$$ F = \frac{\text{Varians Antar-Kelompok}}{\text{Varians Dalam-Kelompok}}$$

Fitur dengan nilai F tertinggi memiliki pengaruh paling besar terhadap variabel target.

  

Hasil uji ini digunakan untuk mengurangi fitur-fitur yang tidak signifikan, seperti `YearsSinceLastPromotion`, `DailyRate`, `PercentSalaryHike`, `DistanceFromHome`, `NumCompaniesWorked`, `HourlyRate`, dan `MonthlyRate`.

  

---

  

## Model Development

  

Model Machine Learning yang Digunakan:

1.  **Logistic Regression:** (default parameter)

Menggunakan fungsi sigmoid untuk prediksi probabilitas:

$$ P(Y=1  \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}} $$

2.  **Random Forest:** ('max_depth=4', 'random_state=0')

Algoritma ansambel yang memanfaatkan agregasi dari banyak pohon keputusan untuk menghasilkan prediksi yang lebih stabil.

3.  **XGBoost (Extreme Gradient Boosting):** ('learning_rate=0.01', 'max_depth=3', 'n_estimators=1000')

Algoritma boosting berbasis pohon keputusan. XGBoost mengoptimalkan fungsi loss seperti *log-loss* menggunakan pendekatan gradient boosting.

  

---

  

## Model Evaluation

  

### Metrik Evaluasi:

1.  **Accuracy:** Mengukur seberapa sering prediksi model benar:

  

$$  \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

2.  **Precision:** Proporsi prediksi positif yang benar

$$  \text{Precision} = \frac{TP}{TP + FP} $$

3.  **Recall (Sensitivity):** Kemampuan model mendeteksi kelas positif

$$  \text{Recall} = \frac{TP}{TP + FN} $$

4.  **F1-Score:** Kombinasi harmonis antara precision dan recall

$$  \text{F1-Score} = \frac{\text  2  \cdot{Precision} \cdot  \text{Recall}}{\text{Precision} + \text{Recall}} $$

5.  **ROC-AUC:**

- Kurva ROC mengukur trade-off antara sensitivitas (True Positive Rate) dan spesifisitas (False Positive Rate), membantu menggambarkan kemampuan model dalam membedakan antara kelas.

  

### **Confusion Matrix:**
Matriks ini membantu memvisualisasikan performa model:

|                     | **Predicted Positive** | **Predicted Negative** |
|---------------------|------------------------|------------------------|
| **Actual Positive**  | True Positive (TP)      | False Negative (FN)    |
| **Actual Negative**  | False Positive (FP)     | True Negative (TN)     |

---

### **Hasil Evaluasi Model untuk Prediksi Attrition Karyawan**

#### Ringkasan Evaluasi

| **Model**             | **Precision** | **Recall** | **F1-Score** | **Akurasi** |
|-----------------------|---------------|------------|--------------|-------------|
| **XGBoost**           | 0.85          | 0.85       | 0.85         | 85%         |
| **Random Forest**     | 0.79          | 0.79       | 0.79         | 79%         |
| **Logistic Regression** | 0.77        | 0.77       | 0.77         | 77%         |


-  **XGBoost:**

![cm_xgboost](https://images2.imgbox.com/65/ea/CnLTbZ12_o.png) ![XGBoost](https://images2.imgbox.com/42/7b/jKimo6R9_o.png)

-  **Random Forest:**

![cm_randomforest](https://images2.imgbox.com/22/10/qLyfHXGS_o.png) ![randomforest](https://images2.imgbox.com/6a/bb/i3bpDCDo_o.png)

-  **Logistic Regression:**

![cm_logress](https://images2.imgbox.com/4f/d9/aUPG1RoU_o.png) ![logress](https://images2.imgbox.com/29/98/BNfhurdT_o.png)

---

  

### Penjelasan Hasil

  

1.  **XGBoost**:

-  **XGBoost** memiliki skor tertinggi pada semua metrik: precision, recall, F1-score, dan akurasi.

- Dengan akurasi 85%, model ini mampu memprediksi karyawan yang "Keluar" maupun "Bertahan" dengan sangat baik tanpa bias signifikan terhadap salah satu kelas.

-  - AUC tertinggi (0.92), menunjukkan kemampuan prediksi yang sangat baik.

  

2.  **Random Forest**:

- Performa model ini berada di tingkat menengah, dengan semua metrik bernilai 79%.

- Meskipun dapat menangani prediksi dengan cukup baik, performanya masih di bawah **XGBoost**.

  

3.  **Logistic Regression**:

- Model ini memiliki skor terendah pada semua metrik, dengan akurasi hanya 77%.

-  **Logistic Regression** kesulitan menangkap hubungan kompleks antar fitur dalam dataset, sehingga menghasilkan prediksi yang kurang optimal.

  

---

  

## Kesimpulan

  

Berdasarkan evaluasi:

-  **XGBoost** adalah model terbaik untuk prediksi attrition karyawan. Model ini memberikan akurasi tinggi serta precision dan recall yang seimbang, sehingga cocok untuk implementasi di dunia nyata.

-  **Random Forest** dapat digunakan sebagai alternatif dengan performa moderat.

-  **Logistic Regression** tidak direkomendasikan karena kurang mampu menangani kompleksitas dataset.

  
  
  

---

## Kesimpulan

  

Berdasarkan hasil evaluasi:

  

#### Faktor Utama Attrition:

- Gaji rendah, usia muda, dan kepuasan kerja rendah adalah penyebab utama pergantian karyawan.

  

#### Model Terbaik:

-  **XGBoost** memberikan performa terbaik berdasarkan evaluasi AUC dan F1-Score. Model ini memberikan akurasi tinggi serta precision dan recall yang seimbang, sehingga cocok untuk implementasi di dunia nyata.

-  **Random Forest** dapat digunakan sebagai alternatif dengan performa moderat.

-  **Logistic Regression** tidak direkomendasikan karena kurang mampu menangani kompleksitas dataset.

  

#### Rekomendasi Bisnis:

1. Tingkatkan kepuasan kerja melalui pelatihan dan promosi internal.

2. Berikan insentif keuangan yang lebih kompetitif kepada karyawan dengan pendapatan rendah.

3. Fokus pada retensi karyawan usia muda, terutama di departemen dengan tingkat attrition tinggi seperti Sales dan HR.

  

---

## Referensi

[1] B. Peng, “Statistical analysis of employee retention,” in Proc.SPIE, International Conference on Statistics, Applied Mathematics, and Computing Science (CSAMCS 2021), Apr. 2022, p. 1216303. doi: 10.1117/12.2628107.

[2] Dr. A. B. Dr. Ayesha Banu, “Graphical Exploratory Data Analysis (GEDA): A Case Study on Employee Attrition,” Journal of Science and Technology, vol. 7, no. 9, pp. 1–11, 2022, doi: 10.46243/jst.2022.v7.i09.pp01-11.

[3] A. Indrawati, “Penerapan Teknik Kombinasi Oversampling Dan Undersampling Hybrid Oversampling and Undersampling Techniques To Handling Imbalanced Dataset,” JIKO(Jurnal Informatika dan Komputer), vol. 4, no. 1, pp. 38–43, 2021, doi: 10.33387/jiko.

[4] S. Bagui and K. Li, “Resampling imbalanced data for network intrusion detection datasets,” J Big Data, vol. 8, no. 1, 2021, doi: 10.1186/s40537-020-00390-x.