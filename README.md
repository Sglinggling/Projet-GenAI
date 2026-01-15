# CineMatch - Assistant Cinéma par IA Générative

**CineMatch** est un moteur de recommandation de films intelligent combinant **Recherche Sémantique (SBERT)** et **IA Générative (LLM)**. Contrairement aux filtres classiques, il comprend le sens de votre demande ("Je veux un thriller psychologique sombre...") pour trouver les meilleurs films et génère des critiques personnalisées.

Ce projet a été réalisé dans le cadre du cours "Data Engineering & AI".

## Fonctionnalités Clés

* **Recherche Sémantique (RAG)** : Utilisation de `Sentence-BERT` pour transformer la demande utilisateur et les résumés de films en vecteurs.
* **Architecture Optimisée** : Système de **Base Vectorielle locale (Pickle)**. Les vecteurs sont pré-calculés, ce qui rend la recherche instantanée (< 50ms) au lieu de recalculer à chaque requête.
* **Génération Augmentée** : Utilisation d'un LLM local (**Ollama**) pour analyser le profil utilisateur et justifier les recommandations.
* **Interface Interactive** : Dashboard **Streamlit** avec graphiques **Plotly** (Radar de pertinence zoomé, Cartes interactives).
* **Caching** : Système de cache intelligent pour limiter les appels API.

## Architecture Technique

Le projet suit une architecture modulaire :

* `src/data` :
    * `build_vectors.py` : Script d'ingestion et de calcul des embeddings (à lancer une fois).
    * `load_data.py` / `preprocess.py` : Nettoyage des données IMDB.
* `src/nlp` : Gestion du modèle d'embedding (SBERT `all-MiniLM-L6-v2`).
* `src/genai` : Client API pour Ollama, gestion du cache et Prompts.
* `src/ui` : Interface utilisateur Streamlit.

## Installation

1.  **Cloner le projet :**
    ```bash
    git clone [https://github.com/Sglinggling/Projet-GenAI.git](https://github.com/Sglinggling/Projet-GenAI.git)
    cd Projet-GenAI
    ```

2.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Générer la base vectorielle (Obligatoire) :**
    Cette étape va créer le fichier optimisé `movies_with_embeddings.pkl`.
    ```bash
    python src/data/build_vectors.py
    ```

4.  **Configurer Ollama (Local) :**
    * Assurez-vous qu'Ollama tourne en local (`http://localhost:11434`).
    * Modèle par défaut : `llama3.2:latest`.

## Utilisation

Une fois les vecteurs générés, lancez l'application :

```bash
streamlit run src/ui/app.py