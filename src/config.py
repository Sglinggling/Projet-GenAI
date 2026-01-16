import os

def configure_hf_ssl_bypass():
    # Désactive la vérification SSL pour les requêtes Hugging Face
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"