"""
# importuje klasę z modułu custom_kmeans.py i udostępnia ją na poziomie
pakietu 'kmeans'
"""

from .custom_kmeans import CustomKMeans

# definiowanie, co ma być eksportowane
__all__ = [
    "CustomKMeans"
]