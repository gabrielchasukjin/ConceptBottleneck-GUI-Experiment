import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_concept_activations(activations, index, concept_names):
    plt.figure(figsize=(10, 6))
    sns.heatmap(activations[index].reshape(1, -1), annot=True, fmt=".2f", cmap="viridis", xticklabels=concept_names, cbar=True)
    plt.title(f'Concept Activations for Sample {index}')
    plt.show()

def plot_concept_contributions(contributions, index, concept_names):
    plt.figure(figsize=(10, 6))
    sns.heatmap(contributions[index].reshape(1, -1), annot=True, fmt=".2f", cmap="coolwarm", xticklabels=concept_names, cbar=True)
    plt.title(f'Concept Contributions for Sample {index}')
    plt.show()
