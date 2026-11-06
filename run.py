"""
Ponto de entrada da aplicação Flask (briefing seção 11).

Uso:
    python run.py
"""
from app.presentation import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
