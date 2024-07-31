
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, make_scorer, precision_score, recall_score

# Excel dosyasını yükle
df = pd.read_excel(r'C:\Users\h_has\Downloads\winequality-red.xlsx')

# Veriyi özellikler (X) ve hedef (y) olarak ayır
X = df.drop('quality', axis=1)  # 'quality' sütunu hedef değişken olduğu varsayılmaktadır
y = df['quality']

# Veriyi eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sınıflandırıcıları eğit
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train_scaled, y_train)

# Tahminler ve confusion matrisleri
classifiers = {'Random Forest': rf_classifier, 'k-NN': knn_classifier}
for name, classifier in classifiers.items():
    y_pred = classifier.predict(X_test_scaled)
    print(f"{name} Sınıflandırıcı")
    print("Confusion Matrisi:\n", confusion_matrix(y_test, y_pred))

# Performans metriklerini hesapla
precision_scorer = make_scorer(precision_score, average='macro', zero_division=1)
recall_scorer = make_scorer(recall_score, average='macro', zero_division=1)

performance_metrics = {'Classifier': [], 'Accuracy': [], 'Precision': [], 'Recall': []}

for name, classifier in classifiers.items():
    accuracy = cross_val_score(classifier, X, y, cv=10, scoring='accuracy').mean()
    precision = cross_val_score(classifier, X, y, cv=10, scoring=precision_scorer).mean()
    recall = cross_val_score(classifier, X, y, cv=10, scoring=recall_scorer).mean()
    
    performance_metrics['Classifier'].append(name)
    performance_metrics['Accuracy'].append(accuracy)
    performance_metrics['Precision'].append(precision)
    performance_metrics['Recall'].append(recall)

# Performans metriklerini içeren DataFrame'i oluştur
performance_df = pd.DataFrame(performance_metrics)

# Tabloyu göster
print(performance_df)
# Kullanıcıdan şarap özelliklerini girmesini iste
user_input = input("Lütfen şarap özelliklerini virgülle ayrılmış olarak girin (örnek: 7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4): ")

# Girdiyi float değerlerin bir listesine dönüştür
new_sample = [float(value) for value in user_input.split(",")]

# Listeyi NumPy dizisine çevir ve ölçeklendir
new_sample_scaled = scaler.transform([new_sample])

# Her sınıflandırıcı için yeni örneğin sınıfını tahmin et ve yazdır
print("\nYeni örneğin tahminleri:")
for name, classifier in classifiers.items():
    prediction = classifier.predict(new_sample_scaled)
    print(f"{name} sınıflandırıcısının tahmini: Kalite Sınıfı {prediction[0]}")
    