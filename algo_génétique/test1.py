import random
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline


# 1. Charger des données d'exemple (par exemple, le dataset 20 newsgroups)
data = fetch_20newsgroups(subset='train', categories=['rec.autos', 'rec.sport.baseball', 'sci.med', 'sci.space'])
X, y = data.data, data.target

# 2. Prétraitement des données et conversion en TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(X)

# Séparation en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# 3. Créer l'algorithme génétique pour optimiser les hyperparamètres du modèle SVM
# Définition de l'algorithme génétique : une classe "Fitness" pour évaluer le modèle et optimiser ses paramètres

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Fonction pour initialiser les individus (optimiser les paramètres C et gamma pour SVM)
def create_individual():
    C = random.uniform(0.1, 10)  # Plage pour C
    gamma = random.uniform(0.001, 0.1)  # Plage pour gamma
    return [C, gamma]

# Fonction d'évaluation : entraîne un modèle SVM et évalue son score sur l'ensemble de test
def evaluate(individual):
    C, gamma = individual
    model = make_pipeline(TfidfVectorizer(max_features=1000), SVC(C=C, gamma=gamma))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred),

# Enregistrer la fonction d'évaluation pour l'algorithme génétique
def main():
    # Créer l'environnement de l'algorithme génétique
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Croisement
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Mutation
    toolbox.register("select", tools.selTournament, tournsize=3)  # Sélection

    # Créer une population initiale
    population = toolbox.population(n=10)

    # Algorithme génétique - évolution sur 10 générations
    for gen in range(10):
        print(f"Generation {gen}")
        
        # Évaluation des individus
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Sélection des meilleurs individus
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Appliquer les opérateurs génétiques
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:  # Taux de croisement
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:  # Taux de mutation
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Réévaluation des individus mutants ou croisés
        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_individuals))
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit
        
        # Remplacer la population précédente par la nouvelle
        population[:] = offspring

        # Afficher la meilleure solution de la génération actuelle
        best_individual = tools.selBest(population, 1)[0]
        print(f"Best individual: {best_individual}, Fitness: {best_individual.fitness.values[0]}")

    # Afficher la meilleure solution finale
    best_individual = tools.selBest(population, 1)[0]
    print(f"Best individual: {best_individual}, Fitness: {best_individual.fitness.values[0]}")

    return best_individual

if __name__ == "__main__":
    best_individual = main()
    # Entraînement final du modèle avec les meilleurs paramètres trouvés
    C, gamma = best_individual
    model = make_pipeline(TfidfVectorizer(max_features=1000), SVC(C=C, gamma=gamma))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Final model accuracy: {accuracy_score(y_test, y_pred)}")
