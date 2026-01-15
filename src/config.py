import requests
from huggingface_hub import configure_http_backend

def configure_hf_ssl_bypass():
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=backend_factory)
