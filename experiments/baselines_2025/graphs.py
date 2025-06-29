import matplotlib.pyplot as plt
import numpy as np

def plot_models_accuracy(models_data):
    """
    Affiche un histogramme des précisions des modèles.
    
    Args:
        models_data (list): Liste de dictionnaires contenant 'name' et 'accuracy' pour chaque modèle.
                         Exemple: [{"name": "Modèle 1", "accuracy": 0.85}, ...]
    """
    # Extraire les noms et les précisions
    model_names = [model["name"] for model in models_data]
    accuracies = [model["accuracy"] for model in models_data]
    
    # Créer la figure avec une hauteur plus grande pour les étiquettes
    plt.figure(figsize=(14, 8))
    
    # Créer les barres avec une largeur réduite pour plus d'espace
    bars = plt.bar(model_names, accuracies, color='skyblue', edgecolor='black', width=0.6)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Ajouter les titres et les labels
    plt.title('Précision des modèles', fontsize=14, pad=20)
    plt.xlabel('Modèles', fontsize=12)
    plt.ylabel('Précision', fontsize=12)
    # Ajuster la rotation et l'alignement des étiquettes
    plt.xticks(rotation=25, ha='right', fontsize=10)
    
    # Ajuster les marges pour éviter que les étiquettes ne soient coupées
    plt.subplots_adjust(bottom=0.25)
    # Ajuster l'échelle Y pour les pourcentages (0-100%)
    plt.ylim(30, 55)  # Ajusté pour les valeurs de précision en pourcentage
    plt.yticks(range(30, 56, 5))  # Graduations tous les 5%
    
    # Ajouter une grille
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajuster la mise en page et afficher
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # Données d'exemple
    models_data = [
        {"name": "KNN CSP", "accuracy": 50},
        {"name": "fuzzy", "accuracy": 49.61},
        {"name": "KNN brute", "accuracy": 48.2},
        {"name": "Logistic Regression", "accuracy": 47.5},
        {"name": "SVM", "accuracy": 44.3},
        {"name": "EEG Net", "accuracy": 42},
        {"name": "Random Forest", "accuracy": 35.89},
    ]
    
    plot_models_accuracy(models_data)