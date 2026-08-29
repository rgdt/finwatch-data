#!/usr/bin/env python3
"""
Construit les listes de tickers surveillés (EU et US) à partir des tables
d'indices de Wikipédia, et les écrit dans universe/eu.txt et universe/us.txt.

Tourne sur un runner GitHub Actions, qui a un accès internet complet.

Chaque source est isolée : si Wikipédia change une table ou tombe, la source
concernée est ignorée avec un avertissement et le reste continue. Si le
résultat est vide ou anormalement petit, le fichier existant est conservé —
mieux vaut un univers légèrement périmé qu'un univers vide.

Ce script n'a pas besoin de tourner à chaque snapshot : une fois par semaine
suffit largement.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_DIR = ROOT / "universe"

# Un ticker Yahoo se compose du symbole local et d'un suffixe de place.
# `suffix` vaut "" pour les places américaines (pas de suffixe chez Yahoo).
SOURCES = {
    "eu": [
        ("CAC 40", "https://en.wikipedia.org/wiki/CAC_40", ".PA"),
        ("SBF 120", "https://fr.wikipedia.org/wiki/SBF_120", ".PA"),
        ("DAX", "https://en.wikipedia.org/wiki/DAX", ".DE"),
        ("MDAX", "https://en.wikipedia.org/wiki/MDAX", ".DE"),
        ("SDAX", "https://en.wikipedia.org/wiki/SDAX", ".DE"),
        ("AEX", "https://en.wikipedia.org/wiki/AEX_index", ".AS"),
        ("BEL 20", "https://en.wikipedia.org/wiki/BEL20", ".BR"),
        ("IBEX 35", "https://en.wikipedia.org/wiki/IBEX_35", ".MC"),
        ("FTSE MIB", "https://en.wikipedia.org/wiki/FTSE_MIB", ".MI"),
    ],
    "us": [
        ("S&P 500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ""),
        ("Nasdaq-100", "https://en.wikipedia.org/wiki/Nasdaq-100", ""),
        ("S&P MidCap 400", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", ""),
        ("S&P SmallCap 600", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", ""),
    ],
}

# Noms de colonnes qui contiennent un symbole boursier, selon la langue et
# les conventions de chaque page.
TICKER_COLUMNS = (
    "ticker", "symbol", "ticker symbol", "code", "code isin", "mnémonique",
    "symbole", "epic",
)

# Un symbole plausible : lettres, chiffres, point et tiret, 1 à 8 caractères.
VALID = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,7}$")

# En deçà de ce seuil, on considère que la construction a échoué et on
# conserve le fichier précédent.
MIN_EXPECTED = {"eu": 150, "us": 800}


def pick_ticker_column(df: pd.DataFrame) -> pd.Series | None:
    """Retourne la colonne du tableau qui contient les symboles, ou None."""
    for col in df.columns:
        name = str(col).strip().lower()
        # Wikipédia produit parfois des colonnes multi-niveaux
        name = re.sub(r"\s+", " ", name.split(".")[0])
        if any(name == c or name.startswith(c) for c in TICKER_COLUMNS):
            return df[col]
    return None


def clean(raw: object, suffix: str) -> str | None:
    """Normalise une cellule en ticker Yahoo, ou None si inexploitable."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    sym = str(raw).strip().upper()
    # Retire les notes de bas de page ("AIR[1]") et les espaces internes
    sym = re.sub(r"\[.*?\]", "", sym).replace(" ", "")
    if not sym:
        return None
    # Certaines tables donnent déjà le ticker suffixé
    if "." in sym and suffix and sym.endswith(suffix):
        pass
    elif suffix:
        sym = sym.split(".")[0] + suffix
    else:
        # Yahoo utilise "-" là où S&P utilise "." pour les classes d'actions
        sym = sym.replace(".", "-")
    return sym if VALID.match(sym) else None


def harvest(name: str, url: str, suffix: str) -> set[str]:
    """Extrait les tickers d'une page d'indice. Ne lève jamais."""
    try:
        tables = pd.read_html(url)
    except Exception as exc:  # réseau, page modifiée, table absente
        print(f"  ! {name}: lecture impossible ({exc.__class__.__name__}: {exc})",
              file=sys.stderr)
        return set()

    found: set[str] = set()
    for table in tables:
        # Aplatit les en-têtes multi-niveaux
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [" ".join(str(p) for p in col).strip()
                             for col in table.columns]
        column = pick_ticker_column(table)
        if column is None:
            continue
        for cell in column:
            ticker = clean(cell, suffix)
            if ticker:
                found.add(ticker)
        # La première table exploitable est presque toujours la bonne
        if len(found) >= 15:
            break

    print(f"  · {name}: {len(found)} tickers")
    return found


def build(region: str) -> None:
    print(f"[{region.upper()}] construction de l'univers")
    tickers: set[str] = set()
    for name, url, suffix in SOURCES[region]:
        tickers |= harvest(name, url, suffix)

    target = UNIVERSE_DIR / f"{region}.txt"
    minimum = MIN_EXPECTED[region]

    if len(tickers) < minimum:
        if target.exists():
            print(f"[{region.upper()}] seulement {len(tickers)} tickers "
                  f"(< {minimum}) — fichier existant conservé", file=sys.stderr)
            return
        print(f"[{region.upper()}] seulement {len(tickers)} tickers "
              f"(< {minimum}) et aucun fichier existant — écriture quand même",
              file=sys.stderr)

    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(sorted(tickers)) + "\n", encoding="utf-8")
    print(f"[{region.upper()}] {len(tickers)} tickers écrits dans {target.name}")


def main() -> int:
    regions = sys.argv[1:] or ["eu", "us"]
    for region in regions:
        if region not in SOURCES:
            print(f"région inconnue : {region}", file=sys.stderr)
            return 2
        build(region)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
