import numpy as np
import pytest
import time
from kmeans.custom_kmeans import CustomKMeans

# importy potrzebne do testów na rzeczywistych danych
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score, confusion_matrix

# import implementacji referencyjnej do porównania
from sklearn.cluster import KMeans as SKLearnKMeans


@pytest.fixture
def sample_data():
    # tworzy prosty zbiór danych dla testów
    X = np.array([
        [1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]
    ])
    return X


# funkcja pomocnicza do mapowania etykiet
def map_kmeans_labels_to_true_labels(y_true, y_pred):
    """
    ponieważ K-Means przydziela etykiety losowo (np. klaster 0 może oznaczać klasę 1),
    ta funkcja mapuje etykiety przewidziane na najbardziej prawdopodobne etykiety prawdziwe
    """
    y_mapped = np.zeros_like(y_pred)
    # dla każdego klastra znalezionego przez K-Means
    for cluster_id in np.unique(y_pred):
        mask = (y_pred == cluster_id)
        # sprawdzamy, jakie prawdziwe etykiety wpadły do tego klastra
        true_labels_in_cluster = y_true[mask]

        if len(true_labels_in_cluster) > 0:
            # Znajdź najczęściej występującą prawdziwą etykietę w tym klastrze (moda)
            most_common_true_label = np.bincount(true_labels_in_cluster).argmax()
            y_mapped[mask] = most_common_true_label
    return y_mapped


# standardowe testy jednostkowe

def test_initialization_shape(sample_data):
    # testuje, czy kształt centroidów po inicjalizacji jest poprawny
    kmeans = CustomKMeans(n_clusters=3, random_state=42)
    kmeans._initialize_centroids(sample_data)
    # powinny być 3 centroidy, każdy z 2 cechami
    assert kmeans.centroids.shape == (3, 2)


def test_fit_labels_shape(sample_data):
    # testuje, czy po dopasowaniu rozmiar etykiet zgadza się z liczbą próbek
    kmeans = CustomKMeans(n_clusters=2, max_iter=1)
    kmeans.fit(sample_data)
    # powinno być 7 etykiet, po jednej dla każdej próbki
    assert kmeans.labels.shape == (sample_data.shape[0],)


def test_convergence(sample_data):
    # testuje, czy algorytm konwerguje i czy centroidy są stabilne
    kmeans = CustomKMeans(n_clusters=2, random_state=1)
    kmeans.fit(sample_data)

    # uruchomienie ponownie z większą liczbą iteracji
    kmeans_retest = CustomKMeans(n_clusters=2, random_state=1, max_iter=20)
    kmeans_retest.fit(sample_data)

    # centroidy powinny być takie same, jeśli konwergencja jest poprawna
    assert np.allclose(kmeans.centroids, kmeans_retest.centroids)


def test_predict_function(sample_data):
    # testuje funkcję predict
    kmeans = CustomKMeans(n_clusters=2, random_state=42)
    kmeans.fit(sample_data)

    # nowa próbka (blisko [1,1])
    new_data = np.array([[1.1, 1.1]])
    prediction = kmeans.predict(new_data)

    assert prediction.shape == (1,)
    # sprawdzanie, czy etykieta jest poprawnym indeksem klastra (0 lub 1)
    assert prediction[0] in [0, 1]


def test_basic_clustering_logic():
    """
    testuje, czy K-Means poprawnie klasteryzuje dwa wyraźnie oddzielone punkty

    powinno to być deterministyczne i osiągnąć konwergencję w jednej iteracji


    dane z dwoma bardzo odległymi punktami
    klaster 1: [10, 10]
    klaster 2: [0, 0]
    """
    X_simple = np.array([
        [10.0, 10.0],
        [10.1, 10.1],
        [9.9, 9.9],
        [0.0, 0.0],
        [0.1, 0.1],
        [-0.1, -0.1]
    ])


    """
    K=2
    ustawiamy random_state tak, aby centroidy były idealnie rozdzielone na początku
    
    własna implementacja używa losowego wyboru próbek z X
    
    użycie random_state=42 często prowadzi do wyboru jednej próbki z każdego klastra
    """
    kmeans = CustomKMeans(n_clusters=2, max_iter=2, random_state=42)

    """
    inicjalizacja centroidów dla random_state=42 w implementacji
    
    zazwyczaj wybiera dwa punkty, które są daleko od siebie
    
    uruchomienie fit
    """
    kmeans.fit(X_simple)

    """
    oczekujemy, że centroidy będą bardzo blisko średniej każdego z dwóch klastrów:
    średnia klastra 1 (małe wartości): ok. [0, 0]
    średnia klastra 2 (duże wartości): ok. [10, 10]
    """

    # sortujemy centroidy, aby uniknąć problemu zamienionych etykiet (0 i 1)
    sorted_centroids = np.sort(kmeans.centroids, axis=0)

    # sprawdzenie pierwszego centroidu (blisko [0, 0])
    expected_center_1 = np.array([0.0, 0.0])
    # sprawdzenie drugiego centroidu (blisko [10, 10])
    expected_center_2 = np.array([10.0, 10.0])

    # używamy allclose, aby porównać wartości zmiennoprzecinkowe z tolerancją
    assert np.allclose(sorted_centroids[0], expected_center_1, atol=0.1)
    assert np.allclose(sorted_centroids[1], expected_center_2, atol=0.1)

    """
    sprawdzenie, czy konwergencja nastąpiła szybko
    
    w tym przypadku nie weryfikujemy dokładnej liczby iteracji, ale stabilność
    """
    assert kmeans.centroids.shape == (2, 2)


# testy walidacyjne na rzeczywistych danych (z liczeniem błędów)

def test_on_breast_cancer_dataset():
    """
    test na zbiorze Breast Cancer (2 klasy);
    porównuje naszą implementację z implementacją biblioteczną
    """
    X, y_true = load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n--- PORÓWNANIE DLA ZBIORU BREAST CANCER ---")

    # 1. nasza implementacja
    start_time_custom = time.time()
    custom_kmeans = CustomKMeans(n_clusters=2, random_state=42)
    custom_labels_raw = custom_kmeans.fit_predict(X_scaled)
    end_time_custom = time.time()

    custom_time = end_time_custom - start_time_custom

    custom_labels_mapped = map_kmeans_labels_to_true_labels(y_true, custom_labels_raw)
    custom_errors = np.sum(y_true != custom_labels_mapped)
    custom_accuracy = accuracy_score(y_true, custom_labels_mapped)

    print(f"\n[Custom] Czas: {custom_time:.6f} s")
    print(f"[Custom] Błędy: {custom_errors} / {len(y_true)}")
    print(f"[Custom] Dokładność: {custom_accuracy:.4f}")

    # 2. implementacja biblioteczna (scikit-learn)
    # używamy n_init=10 (domyślne), aby porównać z pełną, zoptymalizowaną wersją
    start_time_sklearn = time.time()
    sklearn_kmeans = SKLearnKMeans(n_clusters=2, random_state=42, n_init="auto")
    sklearn_labels_raw = sklearn_kmeans.fit_predict(X_scaled)
    end_time_sklearn = time.time()

    sklearn_time = end_time_sklearn - start_time_sklearn

    sklearn_labels_mapped = map_kmeans_labels_to_true_labels(y_true, sklearn_labels_raw)
    sklearn_errors = np.sum(y_true != sklearn_labels_mapped)
    sklearn_accuracy = accuracy_score(y_true, sklearn_labels_mapped)

    print(f"\n[SKLearn] Czas: {sklearn_time:.6f} s")
    print(f"[SKLearn] Błędy: {sklearn_errors} / {len(y_true)}")
    print(f"[SKLearn] Dokładność: {sklearn_accuracy:.4f}")

    # porównanie
    print(f"\nRóżnica czasu (SKLearn - Custom): {sklearn_time - custom_time:.6f} s")
    print(f"Różnica dokładności (SKLearn - Custom): {sklearn_accuracy - custom_accuracy:.4f}")

    # asercja dla naszej implementacji
    assert custom_accuracy > 0.85, f"Zbyt niska dokładność Custom: {custom_accuracy}"


def test_on_digits_dataset():
    """
    test na zbiorze Digits (10 cyfr);
    porównuje naszą implementację z implementacją biblioteczną
    """
    X, y_true = load_digits(return_X_y=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n--- PORÓWNANIE DLA ZBIORU DIGITS ---")

    # 1. nasza implementacja
    start_time_custom = time.time()
    custom_kmeans = CustomKMeans(n_clusters=10, random_state=42)
    custom_labels_raw = custom_kmeans.fit_predict(X_scaled)
    end_time_custom = time.time()

    custom_time = end_time_custom - start_time_custom

    custom_labels_mapped = map_kmeans_labels_to_true_labels(y_true, custom_labels_raw)
    custom_errors = np.sum(y_true != custom_labels_mapped)
    custom_accuracy = accuracy_score(y_true, custom_labels_mapped)

    print(f"\n[Custom] Czas: {custom_time:.6f} s")
    print(f"[Custom] Błędy: {custom_errors} / {len(y_true)}")
    print(f"[Custom] Dokładność: {custom_accuracy:.4f}")

    # 2. implementacja biblioteczna (scikit-learn)
    start_time_sklearn = time.time()
    sklearn_kmeans = SKLearnKMeans(n_clusters=10, random_state=42, n_init="auto")
    sklearn_labels_raw = sklearn_kmeans.fit_predict(X_scaled)
    end_time_sklearn = time.time()

    sklearn_time = end_time_sklearn - start_time_sklearn

    sklearn_labels_mapped = map_kmeans_labels_to_true_labels(y_true, sklearn_labels_raw)
    sklearn_errors = np.sum(y_true != sklearn_labels_mapped)
    sklearn_accuracy = accuracy_score(y_true, sklearn_labels_mapped)

    print(f"\n[SKLearn] Czas: {sklearn_time:.6f} s")
    print(f"[SKLearn] Błędy: {sklearn_errors} / {len(y_true)}")
    print(f"[SKLearn] Dokładność: {sklearn_accuracy:.4f}")

    # porównanie
    print(f"\nRóżnica czasu (SKLearn - Custom): {sklearn_time - custom_time:.6f} s")
    print(f"Różnica dokładności (SKLearn - Custom): {sklearn_accuracy - custom_accuracy:.4f}")

    # asercja dla naszej implementacji
    assert custom_accuracy > 0.50, f"Zbyt niska dokładność Custom: {custom_accuracy}"
