#!/usr/bin/env python3
"""
Produit le snapshot de marché lu par les briefs Financial Watchtower.

Tourne sur un runner GitHub Actions (accès internet complet), écrit
data/snapshot-<region>.json, écrasé à chaque exécution.

Le script ne décide rien : il mesure. La sélection des 15 actifs, la
répartition en catégories et les lectures par horizon se font en aval, à
partir de ces métriques et de l'actualité. Ici on se contente de fournir des
chiffres justes et datés.

Usage :
    python scripts/fetch_snapshot.py eu
    python scripts/fetch_snapshot.py us
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CONFIG — les seuls réglages à toucher au quotidien
# ---------------------------------------------------------------------------

PRICE_MIN_EUR = 3.50      # cours plancher
PRICE_MAX_EUR = 50.00     # cours plafond
MAX_POSITION_EUR = 50.00  # taille de position maximale, titres entiers

# Ce script ne filtre rien. Il mesure tout l'univers et publie tout.
#
# Le tri, la fourchette de prix, le seuil de liquidité et la sélection sont
# appliqués en aval par `claude/screener.py`, dans le projet — là où vivent
# aussi les positions ouvertes. Le dépôt reste ainsi un pur relais de données :
# il ne sait rien de qui le lit, et changer un critère ne demande ni commit ni
# nouvelle exécution des workflows.
#
# Seul le contexte détaillé (nom, secteur, prochaine date de résultats) reste
# limité aux titres les plus actifs : il coûte un appel réseau par titre, ce qui
# est intenable sur 1 500.
CONTEXT_TOP_N = 80

# Nombre de séances de cours conservées pour les candidats, pour les
# graphiques du brief.
SERIES_DAYS = 90

# Liquidité minimale : en dessous, le carnet est trop mince pour qu'une
# lecture technique ait un sens, même sur des positions de 50 €.
MIN_AVG_VALUE_EUR = 250_000  # volume moyen 20j × cours, en euros

# Poids du volume relatif dans le score d'activité. À 1,0 le volume et la
# performance pèsent pareil ; au-dessus, l'anomalie de volume prime.
VOLUME_EXPONENT = 1.5


BATCH_SIZE = 120          # tickers par appel yfinance
DOWNLOAD_RETRIES = 3
RETRY_BACKOFF = 20        # secondes, doublé à chaque échec

# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


# --- Taux de change ---------------------------------------------------------

FX_PAIRS = {
    "USD": "EURUSD=X",
    "GBP": "EURGBP=X",
    "GBp": "EURGBP=X",   # pence : Yahoo cote certaines valeurs londoniennes ainsi
    "CHF": "EURCHF=X",
    "SEK": "EURSEK=X",
    "DKK": "EURDKK=X",
    "NOK": "EURNOK=X",
    "PLN": "EURPLN=X",
}


def fetch_fx() -> dict[str, float]:
    """Retourne les taux <devise> -> EUR. L'euro vaut 1 par construction."""
    rates = {"EUR": 1.0}
    for currency, pair in FX_PAIRS.items():
        try:
            hist = yf.Ticker(pair).history(period="5d", interval="1d")
            if hist.empty:
                continue
            eur_to_x = float(hist["Close"].iloc[-1])
            if eur_to_x > 0:
                # Yahoo cote EURUSD = combien de USD pour 1 EUR.
                rate = 1.0 / eur_to_x
                # Le pence vaut un centième de livre.
                rates[currency] = rate / 100 if currency == "GBp" else rate
        except Exception as exc:
            log(f"  ! taux {pair} indisponible ({exc.__class__.__name__})")
    log(f"  taux de change récupérés : {sorted(rates)}")
    return rates


# --- Téléchargement des cours ----------------------------------------------


# Traces de fraîcheur, remplies par repair() et publiées dans le snapshot.
FRESHNESS = {
    "raw_last_dates": Counter(),      # dernière date reçue de Yahoo, avant réparation
    "kept_last_dates": Counter(),     # dernière date conservée
    "repaired_bars": 0,               # barres dont les extrêmes ont été reconstruits
    "dropped_last_rows": Counter(),   # champs manquants ayant coûté la dernière séance
}


def repair(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Rend une série exploitable sans lui coûter sa séance la plus récente.

    Yahoo renvoie fréquemment une barre dont la clôture est présente mais dont
    les extrêmes manquent. La supprimer coûtait la séance la plus récente — sur
    un screener de momentum, perdre le dernier jour fausse tout. On répare donc
    plutôt que d'écarter : une barre sans extrêmes se reconstruit à partir de
    l'ouverture et de la clôture, ce qui donne un true range égal à l'écart de
    clôture à clôture. C'est la valeur juste pour une séance dont on ne connaît
    que ces points, pas une invention.

    Seules les barres sans clôture sont écartées : celles-là ne portent
    réellement aucune information.
    """
    if df.empty:
        return df

    raw_last = df.dropna(how="all")
    if not raw_last.empty:
        FRESHNESS["raw_last_dates"][str(raw_last.index[-1].date())] += 1

    close = df["Close"]
    kept = df[close.notna()].copy()

    if len(kept) < len(raw_last):
        # Note ce qui manquait sur la dernière barre écartée, pour diagnostic
        lost = raw_last.index.difference(kept.index)
        if len(lost) and lost[-1] == raw_last.index[-1]:
            row = df.loc[raw_last.index[-1]]
            missing = ",".join(sorted(c for c in ("Open", "High", "Low", "Close", "Volume")
                                      if c in row.index and pd.isna(row[c])))
            FRESHNESS["dropped_last_rows"][missing or "inconnu"] += 1

    if kept.empty:
        return kept

    # Reconstruit les extrêmes manquants à partir des points connus
    for col, agg in (("High", "max"), ("Low", "min")):
        if col not in kept:
            continue
        gaps = kept[col].isna()
        if gaps.any():
            bounds = kept[["Open", "Close"]].agg(agg, axis=1)
            kept[col] = kept[col].fillna(bounds).fillna(kept["Close"])
            FRESHNESS["repaired_bars"] += int(gaps.sum())

    FRESHNESS["kept_last_dates"][str(kept.index[-1].date())] += 1
    return kept


def download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Télécharge l'historique quotidien, par lots, avec reprise sur échec."""
    frames: dict[str, pd.DataFrame] = {}

    for start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[start:start + BATCH_SIZE]
        label = f"{start + 1}-{start + len(batch)}/{len(tickers)}"
        delay = RETRY_BACKOFF

        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                raw = yf.download(
                    batch,
                    period="1y",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                    threads=True,
                )
                break
            except Exception as exc:
                # Yahoo rate-limite régulièrement les runners : on patiente.
                log(f"  ! lot {label} échoué ({exc.__class__.__name__}), "
                    f"tentative {attempt}/{DOWNLOAD_RETRIES}")
                if attempt == DOWNLOAD_RETRIES:
                    raw = None
                    break
                time.sleep(delay)
                delay *= 2

        if raw is None or raw.empty:
            log(f"  ! lot {label} abandonné")
            continue

        for ticker in batch:
            try:
                df = raw[ticker] if len(batch) > 1 else raw
            except KeyError:
                continue

            df = repair(df, ticker)
            if len(df) >= 60:  # trop court pour des moyennes exploitables
                frames[ticker] = df

        log(f"  lot {label} : {len(frames)} séries valides cumulées")
        time.sleep(1)  # courtoisie vis-à-vis de Yahoo

    return frames


# --- Indicateurs ------------------------------------------------------------


def rsi(close: pd.Series, window: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    last_gain, last_loss = gain.iloc[-1], loss.iloc[-1]
    if pd.isna(last_gain) or pd.isna(last_loss) or last_loss == 0:
        return float("nan")
    return float(100 - 100 / (1 + last_gain / last_loss))


def atr_pct(df: pd.DataFrame, window: int = 14) -> float:
    """ATR rapporté au cours : la volatilité quotidienne typique, en %."""
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    value = true_range.rolling(window).mean().iloc[-1]
    close = df["Close"].iloc[-1]
    if pd.isna(value) or close == 0:
        return float("nan")
    return float(value / close * 100)


def pct_change(close: pd.Series, days: int) -> float:
    if len(close) <= days:
        return float("nan")
    past = close.iloc[-1 - days]
    if past == 0 or pd.isna(past):
        return float("nan")
    return float((close.iloc[-1] / past - 1) * 100)


def safe(value: float, digits: int = 2) -> float | None:
    """JSON n'accepte ni NaN ni inf — on les remplace par null."""
    if value is None or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def measure(ticker: str, df: pd.DataFrame, fx: dict[str, float],
            currency: str) -> dict | None:
    """Calcule les métriques d'un titre. Ne juge pas de son éligibilité.

    Retourne None uniquement quand la mesure est impossible : historique trop
    court, devise inconnue, volume nul. Un titre hors fourchette de prix ou peu
    liquide est mesuré comme les autres — c'est au screener, en aval, de
    décider s'il le retient. Une position ouverte passée sous le cours plancher
    est précisément celle dont on veut des nouvelles.
    """
    close = df["Close"].dropna()
    if len(close) < 60:
        return None

    rate = fx.get(currency)
    if rate is None:
        return None

    last = float(close.iloc[-1])
    price_eur = last * rate

    volume = df["Volume"].dropna()
    avg_volume_20 = float(volume.tail(20).mean()) if len(volume) >= 20 else float("nan")
    if pd.isna(avg_volume_20) or avg_volume_20 <= 0:
        return None

    avg_value_eur = avg_volume_20 * price_eur

    last_volume = float(volume.iloc[-1])
    rel_volume = last_volume / avg_volume_20

    prev_close = float(close.iloc[-2])
    sma20 = float(close.tail(20).mean())
    sma50 = float(close.tail(50).mean())
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else float("nan")

    window_52w = close.tail(252)
    high_52w, low_52w = float(window_52w.max()), float(window_52w.min())
    span = high_52w - low_52w
    position_52w = (last - low_52w) / span * 100 if span > 0 else float("nan")

    ret_1d = pct_change(close, 1)
    ret_5d = pct_change(close, 5)
    ret_20d = pct_change(close, 20)

    # Un titre "actif" bouge ET s'échange anormalement. Le score est non signé :
    # il fait remonter les mouvements dans les deux sens, la direction étant
    # portée par les rendements eux-mêmes.
    #
    # L'exposant sur le volume est ce qui empêche le score de n'être qu'une
    # mesure de momentum déguisée : avec log1p, l'écart entre un volume de 0,8
    # et un volume de 2,2 se tassait à un facteur 2, contre 4 pour l'écart de
    # performance — le volume ne pesait presque rien. Élevé à la puissance 1,5,
    # il retrouve une amplitude comparable, et un titre qui monte sur un volume
    # inférieur à sa moyenne est pénalisé au lieu d'être simplement peu
    # récompensé.
    momentum = abs(ret_5d) if not pd.isna(ret_5d) else 0.0
    activity = momentum * max(rel_volume, 0.05) ** VOLUME_EXPONENT

    max_shares = int(MAX_POSITION_EUR // price_eur)

    return {
        "ticker": ticker,
        "currency": currency,
        "close": safe(last, 4),
        "close_eur": safe(price_eur, 4),
        "prev_close": safe(prev_close, 4),
        "date": str(close.index[-1].date()),
        "ret_1d_pct": safe(ret_1d),
        "ret_5d_pct": safe(ret_5d),
        "ret_20d_pct": safe(ret_20d),
        "ret_60d_pct": safe(pct_change(close, 60)),
        "rel_volume": safe(rel_volume),
        "avg_volume_20": safe(avg_volume_20, 0),
        "avg_value_eur": safe(avg_value_eur, 0),
        "atr_pct": safe(atr_pct(df)),
        "rsi_14": safe(rsi(close)),
        "dist_sma20_pct": safe((last / sma20 - 1) * 100),
        "dist_sma50_pct": safe((last / sma50 - 1) * 100),
        "dist_sma200_pct": safe((last / sma200 - 1) * 100) if not pd.isna(sma200) else None,
        "high_52w": safe(high_52w, 4),
        "low_52w": safe(low_52w, 4),
        "position_52w_pct": safe(position_52w),
        "max_shares_50eur": max_shares,
        "capital_deployed_eur": safe(max_shares * price_eur),
        "activity_score": safe(activity, 3),
    }


# --- Enrichissement des candidats ------------------------------------------


def add_context(candidates: list[dict], frames: dict[str, pd.DataFrame]) -> None:
    """Ajoute nom, secteur, prochaine date de résultats et série de cours.

    La date de résultats est ce qui alimente l'horizon 4 h du brief, réservé
    aux catalyseurs événementiels. On ne la demande que pour les candidats
    retenus : un appel par titre, c'est trop lent sur tout l'univers.
    """
    for item in candidates:
        ticker = item["ticker"]
        try:
            info = yf.Ticker(ticker)
            meta = info.get_info() or {}
            item["name"] = meta.get("longName") or meta.get("shortName") or ticker
            item["sector"] = meta.get("sector")
            item["industry"] = meta.get("industry")
            item["market_cap"] = meta.get("marketCap")
            item["exchange"] = meta.get("exchange")

            earnings = None
            calendar = info.calendar or {}
            dates = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
            if dates:
                first = dates[0] if isinstance(dates, (list, tuple)) else dates
                earnings = str(first)
            item["next_earnings"] = earnings
        except Exception as exc:
            log(f"  ! contexte indisponible pour {ticker} "
                f"({exc.__class__.__name__})")
            item.setdefault("name", ticker)
            item.setdefault("next_earnings", None)

        time.sleep(0.3)


def attach_series(rows: list[dict], frames: dict[str, pd.DataFrame]) -> None:
    """Ajoute la série de cours de chaque titre, pour les graphiques du brief.

    Clôtures seules : le volume par séance n'est utilisé nulle part en aval, et
    il pesait un tiers du fichier une fois l'univers entier publié. Les mesures
    de volume qui servent — moyenne 20 séances et volume relatif — sont déjà
    dans les métriques.
    """
    for item in rows:
        df = frames.get(item["ticker"])
        if df is None:
            continue
        tail = df.tail(SERIES_DAYS)
        item["series"] = {
            "dates": [str(d.date()) for d in tail.index],
            "close": [safe(v, 4) for v in tail["Close"]],
        }


# --- Programme principal ----------------------------------------------------


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("eu", "us"):
        print("usage: fetch_snapshot.py <eu|us>", file=sys.stderr)
        return 2
    region = sys.argv[1]

    universe_file = ROOT / "universe" / f"{region}.txt"
    if not universe_file.exists():
        print(f"univers absent : {universe_file}", file=sys.stderr)
        return 1

    tickers = [line.strip() for line in
               universe_file.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#")]
    log(f"[{region.upper()}] univers : {len(tickers)} tickers")

    fx = fetch_fx()

    log(f"[{region.upper()}] téléchargement des cours")
    frames = download(tickers)
    log(f"[{region.upper()}] {len(frames)} séries récupérées")

    if not frames:
        print("aucune série récupérée — snapshot non écrit", file=sys.stderr)
        return 1

    # La devise se déduit du suffixe du ticker, sans appel réseau supplémentaire.
    suffix_currency = {
        ".PA": "EUR", ".AS": "EUR", ".BR": "EUR", ".MC": "EUR",
        ".MI": "EUR", ".DE": "EUR", ".LS": "EUR", ".HE": "EUR",
        ".IR": "EUR", ".VI": "EUR",
        # Places allemandes régionales, courantes chez les courtiers grand public
        ".F": "EUR", ".BE": "EUR", ".HM": "EUR", ".DU": "EUR",
        ".MU": "EUR", ".SG": "EUR",
        ".L": "GBp", ".SW": "CHF", ".ST": "SEK", ".CO": "DKK",
        ".OL": "NOK", ".WA": "PLN",
    }

    measured: list[dict] = []
    for ticker, df in frames.items():
        suffix = "." + ticker.split(".")[-1] if "." in ticker else ""
        currency = suffix_currency.get(suffix, "USD" if region == "us" else "EUR")
        try:
            row = measure(ticker, df, fx, currency)
        except Exception as exc:
            log(f"  ! calcul impossible pour {ticker} ({exc.__class__.__name__})")
            continue
        if row:
            measured.append(row)

    measured.sort(key=lambda r: r["activity_score"] or 0, reverse=True)

    in_band = sum(1 for r in measured
                  if r["close_eur"] and PRICE_MIN_EUR <= r["close_eur"] <= PRICE_MAX_EUR)
    log(f"[{region.upper()}] {len(measured)} titres mesurés, "
        f"dont {in_band} dans la fourchette {PRICE_MIN_EUR}-{PRICE_MAX_EUR} € "
        f"(à titre indicatif : le filtrage se fait en aval)")

    log(f"[{region.upper()}] contexte des {CONTEXT_TOP_N} titres les plus actifs")
    add_context(measured[:CONTEXT_TOP_N], frames)

    log(f"[{region.upper()}] séries de cours")
    attach_series(measured, frames)

    snapshot = {
        "region": region,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Aucun filtrage appliqué. Tri par activity_score décroissant. "
                 "Les critères de sélection vivent dans claude/screener.py, "
                 "côté projet."),
        "fx_to_eur": {k: safe(v, 6) for k, v in fx.items()},
        "universe_size": len(tickers),
        "series_downloaded": len(frames),
        "measured_count": len(measured),
        "context_count": min(CONTEXT_TOP_N, len(measured)),
        # De quand datent réellement les cours. Une séance manquante ne se voit
        # pas en lisant les chiffres — elle se lit ici.
        "freshness": {
            "last_session": (FRESHNESS["kept_last_dates"].most_common(1) or [(None, 0)])[0][0],
            "raw_last_dates": dict(FRESHNESS["raw_last_dates"].most_common(4)),
            "kept_last_dates": dict(FRESHNESS["kept_last_dates"].most_common(4)),
            "repaired_bars": FRESHNESS["repaired_bars"],
            "dropped_last_rows": dict(FRESHNESS["dropped_last_rows"]),
        },
        "instruments": measured,
    }

    out = ROOT / "data" / f"snapshot-{region}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    log(f"[{region.upper()}] écrit : {out.relative_to(ROOT)} "
        f"({out.stat().st_size / 1024:.0f} Ko)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
