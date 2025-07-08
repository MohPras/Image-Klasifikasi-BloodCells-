# Proyek Image Klasifikasi BloodCells

![image](https://github.com/user-attachments/assets/1086f7dc-96be-47e9-b49f-3762867635e6)

----------------------------------

## 🧬 **Latar Belakang**

Sel darah manusia memainkan peran vital dalam menjaga fungsi tubuh, termasuk sistem kekebalan, pengangkutan oksigen, dan pembekuan darah. Pemeriksaan sel darah, terutama melalui **analisis mikroskopis**, merupakan salah satu metode paling umum dalam diagnosa medis, seperti **anemia, infeksi, hingga kelainan darah seperti leukemia**. Namun, proses ini sangat bergantung pada **pengamatan manual oleh ahli patologi**, yang memakan waktu, rentan terhadap subjektivitas, dan membutuhkan keahlian tinggi.

Seiring dengan berkembangnya teknologi **kecerdasan buatan (AI)** dan **pengolahan citra medis**, penggunaan **machine learning** dan **deep learning** untuk membantu klasifikasi sel darah menjadi semakin relevan. Khususnya, **Convolutional Neural Network (CNN)** terbukti sangat efektif dalam menangani data visual seperti citra mikroskopis, karena kemampuannya mengekstraksi fitur penting dari gambar tanpa perlu rekayasa fitur manual.

Dalam proyek ini, dilakukan pengembangan model klasifikasi gambar sel darah tepi normal menggunakan pendekatan **deep learning berbasis CNN**. Dataset yang digunakan terdiri dari 17.092 gambar sel darah dengan delapan kategori utama, seperti neutrofil, eosinofil, limfosit, dan lainnya, yang diperoleh dari **Rumah Sakit Klinik Barcelona** dan telah diberi label oleh **ahli patologi klinis**.

Tujuan utama dari proyek ini adalah untuk membangun sistem klasifikasi otomatis yang mampu mengenali jenis sel darah dengan **akurasi tinggi**, sehingga dapat digunakan sebagai **alat bantu diagnosis awal**, mempercepat proses analisis, serta mengurangi beban kerja tenaga medis. Proyek ini juga berfungsi sebagai eksplorasi awal dalam membandingkan performa model CNN pada data citra medis dengan kualitas tinggi.

----------------------------------

## 🧠 **Business Understanding**

### ❗ Problem Statements

* Proses identifikasi jenis sel darah dari citra mikroskopis saat ini masih mengandalkan **pemeriksaan manual oleh ahli patologi**, yang dapat memakan waktu lama dan rentan terhadap **human error**.
* Volume data medis yang tinggi dan kebutuhan akan diagnosis cepat membuat sistem manual menjadi **tidak efisien** untuk kebutuhan laboratorium skala besar.
* Tidak adanya sistem otomatis yang akurat dalam mengenali berbagai jenis **sel darah normal**, menghambat potensi **otomatisasi** dan **pemetaan awal kondisi darah pasien**.

### 🎯 Goals

* Mengembangkan model klasifikasi gambar berbasis **Convolutional Neural Network (CNN)** untuk mengenali **jenis-jenis sel darah normal** secara otomatis dari citra mikroskopis.
* Mencapai **akurasi klasifikasi yang tinggi** sehingga model dapat diandalkan untuk membantu proses diagnostik awal di laboratorium.
* Membangun fondasi sistem berbasis AI yang dapat diintegrasikan dalam workflow medis untuk mempercepat dan meningkatkan kualitas diagnosis hematologi.

### 💡 Solution Statements

* Menggunakan dataset berkualitas tinggi yang terdiri dari **17.092 gambar sel darah normal**, telah diberi label oleh ahli patologi.
* Melatih model **deep learning CNN** pada data tersebut untuk mengenali dan mengklasifikasikan **8 jenis sel darah tepi**.
* Mengevaluasi performa model menggunakan metrik seperti **akurasi, precision, recall, dan confusion matrix**, serta mengoptimalkan arsitektur dan parameter CNN untuk hasil terbaik.
* Hasil akhir berupa **model klasifikasi otomatis** yang dapat membantu mempercepat analisis sampel darah dan mendukung pengambilan keputusan klinis.

----------------------------------

## 📊 **Data Understanding**

### 📁 Sumber Data

Dataset yang digunakan dalam proyek ini adalah dataset **gambar sel darah tepi (peripheral blood cells)** yang diperoleh dari Laboratorium Inti di **Hospital Clinic of Barcelona**, menggunakan alat otomatis **CellaVision DM96**. Dataset ini bersifat **terbuka (open dataset)** dan telah dianotasi oleh **ahli patologi klinis** profesional.

Dataset tersedia secara publik di Kaggle:
🔗 [Blood Cells Image Dataset – Kaggle](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset)

### 🔍 Deskripsi Dataset

* **Jumlah Total Gambar**: 17.092 gambar
* **Ukuran Gambar**: 360 x 363 piksel
* **Format File**: JPG
* **Jenis Citra**: Mikroskopis, menampilkan **satu sel per gambar**
* **Kategori Sel** (8 kelompok utama):

  1. **Neutrophils**
  2. **Eosinophils**
  3. **Basophils**
  4. **Lymphocytes**
  5. **Monocytes**
  6. **Immature Granulocytes** (promyelocytes, myelocytes, metamyelocytes)
  7. **Erythroblasts**
  8. **Platelets** (Thrombocytes)

### 🧪 Kondisi Data

* Seluruh gambar diambil dari **individu sehat**, **tanpa infeksi**, **tanpa penyakit hematologi/onkologi**, dan **tidak sedang menjalani terapi farmakologis** pada saat pengambilan sampel.
* Dataset mencerminkan kondisi **sel darah normal**, sehingga ideal untuk membangun **model baseline klasifikasi** jenis sel darah sehat.

### 📊 Distribusi Kelas

* Setiap kelas memiliki jumlah gambar yang **bervariasi**, ada kelas yang dominan seperti **neutrophil dan lymphocyte**, serta kelas yang lebih sedikit seperti **basophil atau immature granulocytes**.
* Ketidakseimbangan ini menjadi perhatian khusus dalam **training model**, karena dapat mempengaruhi akurasi klasifikasi terhadap kelas minoritas.

### ⚠️ Potensi Tantangan

* **Ketidakseimbangan data antar kelas** dapat menyebabkan bias model.
* Gambar memiliki **kemiripan visual tinggi antar beberapa jenis sel**, menuntut arsitektur model yang mampu mengekstrak fitur halus.
* Tidak adanya informasi pasien atau metadata lain membuat klasifikasi hanya bergantung pada **citra visual murni**.

----------------------------------

## 🧼 **Data Preprocessing**

Sebelum dilakukan pelatihan model, data melalui beberapa tahap praproses untuk memastikan kualitas input yang optimal dan mencegah overfitting.

### 📊 **Pembagian Dataset**

Dataset dibagi menjadi tiga bagian utama dengan rasio sebagai berikut:

* **64%** untuk **data pelatihan (training set)**
* **16%** untuk **data validasi (validation set)**
* **20%** untuk **data pengujian (test set)**

Pembagian ini dilakukan secara **stratified** untuk menjaga proporsi distribusi kelas yang seimbang di setiap subset data.

### 🧪 **Augmentasi Data (Data Augmentation)**

Untuk meningkatkan keragaman data dan memperkuat kemampuan generalisasi model, dilakukan augmentasi data secara acak selama pelatihan. Teknik augmentasi yang digunakan meliputi:

```python
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.4)
])
```

#### 📌 Penjelasan Teknik:

* `RandomFlip`: Membalik gambar secara acak ke arah **horizontal maupun vertikal**, berguna untuk melatih model agar tidak terlalu sensitif terhadap orientasi gambar.
* `RandomRotation(0.4)`: Memutar gambar secara acak hingga 40% dari sudut penuh (setara dengan ±144 derajat), membantu model mengenali objek dari berbagai sudut pandang.
* Augmentasi ini hanya diterapkan pada **data pelatihan**, sedangkan data validasi dan pengujian tetap dalam bentuk aslinya untuk menjaga konsistensi evaluasi performa model.

----------------------------------


## 🧠 **Modeling**

Dalam tahap ini, dilakukan pembangunan model **Convolutional Neural Network (CNN)** untuk klasifikasi gambar sel darah ke dalam 8 kelas. Model dirancang menggunakan framework **TensorFlow dan Keras**, dengan pendekatan eksperimental yang dicatat dalam bentuk *lab book* untuk memantau performa tiap model.

### 🏗️ Arsitektur Model

Model dibangun menggunakan arsitektur sekuensial (Sequential) dengan beberapa lapisan konvolusi dan *pooling* bertahap. Struktur model adalah sebagai berikut:

```python
model = keras.Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Rescaling(1./255),
    data_augmentation,
    Conv2D(8, (5,5), padding="same", activation="relu"),
    MaxPooling2D(),
    Conv2D(16, (3,3), padding="same", activation="relu"),
    MaxPooling2D(),
    Conv2D(32, (3,3), padding="same", activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(8, activation="softmax")  # Sesuai jumlah kelas
])
```

* **Rescaling** dilakukan untuk menormalkan nilai piksel antara 0 dan 1.
* **Data augmentation** ditambahkan di awal untuk meningkatkan generalisasi.
* Tiga buah lapisan **Conv2D + MaxPooling** digunakan untuk mengekstraksi fitur.
* Output layer menggunakan **softmax** untuk klasifikasi multikelas.

### 🧪 Kompilasi dan Pelatihan

Model dikompilasi dengan konfigurasi berikut:

* **Loss Function**: `SparseCategoricalCrossentropy`
* **Optimizer**: `Adam`
* **Metrik**: `Accuracy`

Model dilatih selama maksimal 100 epoch dengan **early stopping** (patience = 5) untuk mencegah overfitting, berdasarkan performa **val\_accuracy**:

```python
early_cb = keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True,
    monitor="val_accuracy",
    min_delta=0.005
)
```

### 📓 Pencatatan Eksperimen (*Lab Book*)

Untuk mencatat hasil eksperimen, digunakan dictionary bernama `lab_book`:

```python
lab_book[name] = {
    "train_accuracy": train_accuracy,
    "val_accuracy": val_accuracy
}
```

Model dengan nama `"Model_2"` menunjukkan hasil evaluasi sebagai berikut (contoh, sesuaikan dengan hasil aktualmu):

* **Akurasi Data Latih**: `0.95%`
* **Akurasi Data Validasi**: `0.95%`


----------------------------------

## 📈 **Evaluation**

Setelah proses pelatihan selesai, model dievaluasi menggunakan data pengujian untuk mengukur performa generalisasi terhadap data yang belum pernah dilihat sebelumnya. Evaluasi dilakukan dengan menggunakan metrik klasifikasi seperti **precision**, **recall**, dan **F1-score** untuk masing-masing kelas.


### 📊 **Hasil Classification Report**

| Kelas | Precision | Recall | F1-Score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.85      | 0.95   | 0.90     | 237     |
| 1     | 1.00      | 0.96   | 0.98     | 596     |
| 2     | 0.94      | 0.95   | 0.95     | 294     |
| 3     | 0.89      | 0.90   | 0.89     | 602     |
| 4     | 0.97      | 0.95   | 0.96     | 241     |
| 5     | 0.94      | 0.87   | 0.90     | 307     |
| 6     | 0.95      | 0.96   | 0.96     | 679     |
| 7     | 1.00      | 0.99   | 1.00     | 462     |

* **Accuracy keseluruhan**: **95%**
* **Macro average**:

  * Precision: 0.94
  * Recall: 0.94
  * F1-Score: 0.94
* **Weighted average**:

  * Precision: 0.95
  * Recall: 0.95
  * F1-Score: 0.95

### 🧠 **Analisis Evaluasi**

* Model menunjukkan **performansi yang sangat baik secara keseluruhan**, dengan akurasi mencapai **95%** pada data uji.
* Kelas dengan performa **tertinggi** adalah kelas **1 dan 7**, dengan precision dan recall hampir sempurna (≥ 0.99).
* Kelas **5** memiliki nilai recall yang sedikit lebih rendah (**0.87**), menandakan bahwa masih terdapat beberapa sel dari kelas ini yang tidak terdeteksi dengan benar.
* Perbedaan nilai **precision** dan **recall** yang kecil di semua kelas menunjukkan bahwa model tidak hanya akurat, tetapi juga seimbang dalam mendeteksi semua kelas.


### ✅ **Kesimpulan Sementara**

Model CNN yang dikembangkan berhasil mengklasifikasikan gambar sel darah dengan sangat baik, menjadikannya kandidat kuat untuk diimplementasikan sebagai **sistem bantu diagnosis awal** berbasis citra mikroskopis.

----------------------------------

## ✅ Kesimpulan

**Model** menunjukkan hasil yang sangat baik dalam mengklasifikasikan data gambar ke dalam 8 kelas yang berbeda.

<img width="876" alt="image" src="https://github.com/user-attachments/assets/fa76f3aa-337e-4ac3-9495-8c6ffa03dfe7" />


- **Akurasi Model:**
  - Akurasi pada data training: 0.95
  - Akurasi pada data testing: 0.95

- **Overfitting:** Model ini tidak mengalami overfitting, karena tidak ada perbedaan akurasi antara data training dan testing.

- **Kinerja Kelas:**
  - Kelas 7 menunjukkan performa luar biasa, dengan precision dan recall yang mendekati 1.00, yang menunjukkan bahwa model sangat akurat dalam memprediksi kelas ini.
  - Kelas lainnya juga menunjukkan performa yang sangat baik, dengan sebagian besar kelas memiliki precision dan recall yang tinggi (di atas 0.90), khususnya kelas 1, 6, dan 7 yang hampir sempurna.
  - Kelas 5 memiliki sedikit penurunan pada recall (0.87), namun masih tetap berada dalam performa yang dapat diterima.

Kesimpulan Model ini bekerja dengan sangat baik pada sebagian besar kelas, dengan akurasi keseluruhan 95%. Hal ini menunjukkan bahwa model mampu memprediksi dengan sangat tepat pada data yang diberikan, dengan sedikit penurunan pada kelas 5.
