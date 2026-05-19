# =========================================
# PERCOBAAN PRAKTIKUM 7
# JARINGAN SYARAF TIRUAN 2
# =========================================

# 1. Import library yang diperlukan
import tensorflow as tf

print(tf.__version__)
print("TensorFlow berhasil")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================
# 2. Memuat dataset iris dari file CSV
# =========================================

dataset = pd.read_csv(
    'iris.csv',
    header=None,
    sep=','
)

# Menyusun data X (fitur) dan y (label)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# =========================================
# 3. Mengonversi label menjadi numerik
# =========================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)


# =========================================
# 4. Membagi dataset menjadi data latih
#    dan data testing
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================
# 5. Membuat model neural network
# =========================================

model = Sequential([
    Input(shape=X_train.shape[1:]),

    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),

    Dense(3, activation='softmax')
])

# Menampilkan summary model
model.summary()


# =========================================
# 6. Compile model
# =========================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# =========================================
# 7. Training model
# =========================================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# =========================================
# 8. Evaluasi model
# =========================================

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")


# =========================================
# 9. Visualisasi loss dan accuracy
# =========================================

pd.DataFrame(history.history).plot(figsize=(10,6))

plt.title("Training History")

plt.show()


# =========================================
# 10. Prediksi data testing
# =========================================

predictions = model.predict(X_test)

# Mengambil indeks probabilitas tertinggi
predicted_classes = predictions.argmax(axis=1)

print("Prediksi:")
print(predicted_classes)

print("Label Asli:")
print(y_test)


# =========================================
# 11. Confusion Matrix
# =========================================

cm = confusion_matrix(y_test, predicted_classes)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel('Predicted')
plt.ylabel('True')

plt.title('Confusion Matrix')

plt.show()


# =========================================
# 12. Prediksi data baru
# =========================================

def predict_new_data():

    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width = float(input("Masukkan sepal width: "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width = float(input("Masukkan petal width: "))

    # Membuat array data baru
    new_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # Prediksi data baru
    prediction = model.predict(new_data)

    predicted_class = prediction.argmax(axis=1)

    # Mengubah hasil prediksi ke label asli
    predicted_label = label_encoder.inverse_transform(predicted_class)

    print(f"Prediksi kelas: {predicted_label[0]}")


# Menjalankan fungsi prediksi
predict_new_data()