import numpy as np

class CustomKMeans:
    # własna implementacja algorytmu K-Means

    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        """
        inicjalizuje algorytm K-Means

        :param n_clusters: liczba klastrów (K)
        :param max_iter: maksymalna liczba iteracji
        :param random_state: ziarno dla inicjalizacji
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.history = []

    def _initialize_centroids(self, X):
        # inicjalizacja klastrów poprzez losowy wybór próbek z danych
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # wybór unikalnych indeksów
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

    def _create_clusters(self, X):
        # przypisywanie każdej próbki do najbliższego centroidu
        distances = np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)  # zwraca indeks najbliższego centroidu

    def _update_centroids(self, X, cluster_assignments):
        # obliczanie nowych centroidów jako średniej z przypisanych próbek
        new_centroids = np.zeros(self.centroids.shape)
        for k in range(self.n_clusters):
            # Próbki należące do k-tego klastra
            cluster_points = X[cluster_assignments == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                # jeśli klaster jest pusty, centroid pozostaje na poprzedniej pozycji
                new_centroids[k] = self.centroids[k]
        return new_centroids

    def fit(self, X):
        """
        uruchamia algorytm K-Means

        :param X: Dane wejściowe (ndarray numpy), gdzie wiersze to próbki,
        a kolumny to cechy
        """
        X = np.asarray(X)
        self._initialize_centroids(X)
        self.history = [self.centroids.copy()]

        for i in range(self.max_iter):
            # 1. przypisanie
            self.labels = self._create_clusters(X)

            # 2. aktualizacja
            new_centroids = self._update_centroids(X, self.labels)

            # zapis po każdej iteracji
            self.history.append(new_centroids.copy())

            # sprawdzenie konwergencji (jeśli centroidy się nie zmieniły)
            if np.allclose(self.centroids, new_centroids):
                print(f"Konwergencja osiągnięta po {i + 1} iteracjach.")
                break

            self.centroids = new_centroids

        return self

    def predict(self, X):
        """
        przewiduje przypisanie klastra dla nowych próbek

        :param X: Dane wejściowe.
        :return: Etykiety klastrów.
        """
        X = np.asarray(X)
        return self._create_clusters(X)

    def transform(self, X):
        # alias dla predict
        return self.predict(X)

    def fit_predict(self, X):
        # dopasowuje dane i zwraca etykiety klastrów
        self.fit(X)
        return self.labels