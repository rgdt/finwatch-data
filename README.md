# finwatch-data

Instantanés de marché produits par GitHub Actions.

## Contenu

| Chemin | Rôle |
|---|---|
| `scripts/build_universe.py` | Listes de tickers surveillés |
| `scripts/fetch_snapshot.py` | Cours, indicateurs, écriture du snapshot |
| `universe/` | Tickers, un par ligne |
| `data/` | Instantanés JSON, écrasés à chaque exécution |

## Usage

```bash
pip install -r requirements.txt
python scripts/build_universe.py eu us
python scripts/fetch_snapshot.py eu
```

Paramètres en tête de `scripts/fetch_snapshot.py`.
