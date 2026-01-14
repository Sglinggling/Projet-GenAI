# CineMatch - Assistant Cinéma par IA Générative

**CineMatch** est un moteur de recommandation de films intelligent combinant **Recherche Sémantique (SBERT)** et **IA Générative (LLM)**. Contrairement aux filtres classiques, il comprend le sens de votre demande ("Je veux un thriller psychologique sombre...") pour trouver les meilleurs films et génère des critiques personnalisées.

Ce projet a été réalisé dans le cadre du cours "IA Generative" (Projet IA Générative).

## Fonctionnalités Clés

* **Recherche Sémantique (RAG - Retrieval)** : Utilisation de `Sentence-BERT` pour transformer la demande utilisateur et les résumés de films en vecteurs et calculer la similarité (Cosine Similarity).
* **Génération Augmentée (RAG - Generation)** : Utilisation d'un LLM local (**Ollama / Llama 3.2**) pour analyser le profil utilisateur et justifier les recommandations.
* **Interface Interactive** : Dashboard développé avec **Streamlit** et **Plotly** incluant des cartes de proximité sémantique.
* **Optimisation** : Système de **Caching** intelligent pour limiter les appels API et accélérer les réponses.

## Architecture Technique

Le projet suit une architecture modulaire :

* `src/data` : Ingestion et prétraitement des métadonnées IMDB.
* `src/nlp` : Gestion du modèle d'embedding (SBERT `all-MiniLM-L6-v2`).
* `src/genai` : Client API pour Ollama, gestion du cache et Prompts.
* `src/ui` : Interface utilisateur Streamlit.

## Installation

1.  **Cloner le projet :**
    ```bash
    git clone https://github.com/Sglinggling/Projet-GenAI.git
    ```

2.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configurer Ollama (Local) :**
    * Assurez-vous qu'Ollama tourne en local (`http://localhost:11434`).
    * Modèle par défaut : `llama3.2:latest` (modifiable dans `client.py`).

## Utilisation

Lancer l'application Streamlit depuis la racine du projet :

```bash
streamlit run src/ui/app.py