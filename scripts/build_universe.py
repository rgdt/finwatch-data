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

import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

# Wikipédia renvoie 403 aux clients qui ne s'identifient pas. Sa politique
# demande un agent utilisateur descriptif avec un moyen de contact — sans quoi
# pandas.read_html échoue en 403 sur un runner.
USER_AGENT = (
    "finwatch-data/1.0 (https://github.com/rgdt/finwatch-data) "
    "python-urllib"
)
FETCH_RETRIES = 3

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_DIR = ROOT / "universe"

# Un ticker Yahoo se compose du symbole local et d'un suffixe de place.
# `suffix` vaut "" pour les places américaines (pas de suffixe chez Yahoo).
# Sources écartées après vérification page par page — Wikipédia ne porte pas de
# tickers pour celles-ci, inutile de les réinterroger :
#   SBF 120 (fr)     liste de liens, aucun tableau
#   CAC Next 20      pas de tableau de composition, renvoi vers Euronext
#   CAC Mid 60       idem, renvoi vers boursier.com
#   SDAX (en et de)  tableau sans colonne de ticker (logo, nom, branche)
#   PSI-20           tableau sans colonne de ticker
# Conséquence assumée : les mid-caps françaises et allemandes hors MDAX ne sont
# pas couvertes. Il faudrait une autre source de composition pour y remédier.

SOURCES = {
    "eu": [
        ("CAC 40", "https://en.wikipedia.org/wiki/CAC_40", ".PA"),
        ("DAX", "https://en.wikipedia.org/wiki/DAX", ".DE"),
        ("MDAX", "https://en.wikipedia.org/wiki/MDAX", ".DE"),
        ("AEX", "https://en.wikipedia.org/wiki/AEX_index", ".AS"),
        ("BEL 20", "https://en.wikipedia.org/wiki/BEL20", ".BR"),
        ("IBEX 35", "https://en.wikipedia.org/wiki/IBEX_35", ".MC"),
        ("FTSE MIB", "https://en.wikipedia.org/wiki/FTSE_MIB", ".MI"),
        # Vérifiées : colonne « Ticker » présente, tickers déjà suffixés pour
        # Stockholm — clean() sait les reconnaître.
        ("SMI", "https://en.wikipedia.org/wiki/Swiss_Market_Index", ".SW"),
        ("OMX Stockholm 30", "https://en.wikipedia.org/wiki/OMX_Stockholm_30", ".ST"),
        # Oslo : les tickers y sont préfixés de la place (« OSE: MPCC »),
        # clean() sait retirer ce préfixe.
        ("OBX", "https://en.wikipedia.org/wiki/OBX_Index", ".OL"),
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
    # Certaines pages préfixent le ticker de sa place ("OSE: MPCC", "BIT: ENI")
    if ":" in sym:
        sym = sym.rsplit(":", 1)[-1]
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


def fetch_html(url: str) -> str:
    """Récupère une page en s'identifiant correctement, avec reprise."""
    delay = 3
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            request = Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "fr,en;q=0.8",
            })
            with urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception:
            if attempt == FETCH_RETRIES:
                raise
            time.sleep(delay)
            delay *= 2
    return ""


def harvest(name: str, url: str, suffix: str) -> tuple[set[str], str]:
    """Extrait les tickers d'une page d'indice. Ne lève jamais.

    Retourne les tickers et un motif. Le motif est ce qui distingue « la page
    a changé de structure » de « la requête a échoué » — un simple compte à
    zéro ne dit pas laquelle des deux s'est produite, et les deux se réparent
    différemment.
    """
    try:
        # read_html reçoit le HTML déjà téléchargé, pas l'URL : c'est ce qui
        # permet de maîtriser les en-têtes de la requête.
        tables = pd.read_html(io.StringIO(fetch_html(url)))
    except Exception as exc:  # réseau, 403, page modifiée, table absente
        reason = f"lecture impossible ({exc.__class__.__name__}: {exc})"
        print(f"  ! {name}: {reason}", file=sys.stderr)
        return set(), reason

    found: set[str] = set()
    saw_column = False
    for table in tables:
        # Aplatit les en-têtes multi-niveaux
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [" ".join(str(p) for p in col).strip()
                             for col in table.columns]
        column = pick_ticker_column(table)
        if column is None:
            continue
        saw_column = True
        for cell in column:
            ticker = clean(cell, suffix)
            if ticker:
                found.add(ticker)
        # La première table exploitable est presque toujours la bonne
        if len(found) >= 15:
            break

    if found:
        reason = "ok"
    elif saw_column:
        reason = "colonne trouvée mais aucun ticker exploitable"
    else:
        reason = f"aucune colonne de ticker dans les {len(tables)} tableaux"

    print(f"  · {name}: {len(found)} tickers" + ("" if found else f" — {reason}"))
    return found, reason


def build(region: str) -> None:
    print(f"[{region.upper()}] construction de l'univers")
    report_path = UNIVERSE_DIR / f"report-{region}.json"
    previous = {}
    if report_path.exists():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    previous_counts = {k: v.get("count", 0) if isinstance(v, dict) else v
                       for k, v in (previous.get("per_source") or {}).items()}

    tickers: set[str] = set()
    per_source: dict[str, dict] = {}
    regressions: list[str] = []

    for i, (name, url, suffix) in enumerate(SOURCES[region]):
        if i:
            time.sleep(1)  # courtoisie vis-à-vis de Wikipédia
        found, reason = harvest(name, url, suffix)
        per_source[name] = {"count": len(found), "reason": reason}
        tickers |= found

        # Une source qui s'effondre sans disparaître est le cas vicieux : le
        # total reste au-dessus du seuil et rien ne se voit. On compare donc
        # chaque source à son propre passage précédent.
        before = previous_counts.get(name)
        if before and len(found) < before * 0.5:
            regressions.append(f"{name} {before} → {len(found)}")

    # Compte rendu par source, versionné avec l'univers.
    report = {
        "region": region,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_unique": len(tickers),
        "per_source": per_source,
        "empty_sources": sorted(n for n, d in per_source.items() if not d["count"]),
        "regressions": regressions,
    }
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                           encoding="utf-8")

    if report["empty_sources"]:
        print(f"[{region.upper()}] ⚠ sources muettes : "
              f"{', '.join(report['empty_sources'])}", file=sys.stderr)
    if regressions:
        print(f"[{region.upper()}] ⚠ sources en recul : "
              f"{'; '.join(regressions)}", file=sys.stderr)

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
