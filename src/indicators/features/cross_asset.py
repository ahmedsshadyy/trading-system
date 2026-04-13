from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.smt import add_smt_divergence
from src.pipeline_runtime import persist_partitioned_dataset

SUPPORTED_CROSS_ASSET_TIMEFRAMES = frozenset({"H1", "H4"})
FX_SYMBOLS = (
    "AUD_USD",
    "EUR_USD",
    "GBP_USD",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
    "USD_JPY",
    "USD_SEK",
)
CONTEXT_SYMBOLS = FX_SYMBOLS + ("XAU_USD", "USOIL", "DXY")
HORIZONS_BY_TIMEFRAME = {
    "H1": (24, 72, 168),
    "H4": (6, 18, 42),
}
LAG_SCAN_LAGS = {
    "H1": tuple(range(-12, 13)),
    "H4": tuple(range(-6, 7)),
}
LAG_SCAN_PAIRS = (
    ("XAU_USD", "USOIL"),
    ("USOIL", "DXY"),
)
SMT_PARTNERS: dict[str, tuple[tuple[str, int], ...]] = {
    "XAU_USD": (("DXY", -1), ("USD_JPY", 1)),
    "USOIL": (("USD_CAD", -1),),
}
GLOBAL_CONTEXT_SYMBOL = "GLOBAL"
GOLD_OIL_ALIAS = "gold_oil"
OIL_DXY_ALIAS = "oil_dxy"
ALIGNMENT_KEY = "__xasset_align_ts"
PAIR_ALIAS_OVERRIDES = {
    ("USOIL", "DXY"): OIL_DXY_ALIAS,
    ("DXY", "USOIL"): OIL_DXY_ALIAS,
    ("XAU_USD", "USOIL"): GOLD_OIL_ALIAS,
    ("USOIL", "XAU_USD"): GOLD_OIL_ALIAS,
}


def _is_supported_timeframe(timeframe: str) -> bool:
    return timeframe in SUPPORTED_CROSS_ASSET_TIMEFRAMES


def _is_commodity_symbol(symbol: str) -> bool:
    return symbol in {"XAU_USD", "USOIL"}


def _partner_token(symbol: str) -> str:
    return symbol.lower()


def _pair_alias(a: str, b: str) -> str:
    return PAIR_ALIAS_OVERRIDES.get((a, b), f"{a.lower()}_{b.lower()}")


def aligned_timestamp_for_instrument(
    timestamp: pd.Series | pd.Index,
    *,
    instrument: str,
    timeframe: str,
) -> pd.Series:
    ts = pd.Series(pd.to_datetime(timestamp, utc=True, errors="coerce"), copy=False)
    if timeframe == "H4" and _is_commodity_symbol(instrument):
        return ts - pd.Timedelta(hours=1)
    return ts


def add_alignment_key(
    df: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
    key: str = ALIGNMENT_KEY,
) -> pd.DataFrame:
    out = df.copy()
    out[key] = aligned_timestamp_for_instrument(
        out["timestamp"],
        instrument=instrument,
        timeframe=timeframe,
    )
    return out


def _prepare_close_frame(
    frame: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
) -> pd.DataFrame:
    working = add_alignment_key(frame, instrument=instrument, timeframe=timeframe)
    out = working[[ALIGNMENT_KEY, "close"]].copy()
    out = out.rename(columns={ALIGNMENT_KEY: "timestamp", "close": instrument})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", instrument])
    out = out.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    out[instrument] = out[instrument].astype(float)
    return out.reset_index(drop=True)


def _iter_correlation_pairs() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for idx, left in enumerate(FX_SYMBOLS):
        for right in FX_SYMBOLS[idx + 1 :]:
            pairs.append((left, right))
    for fx in FX_SYMBOLS:
        pairs.append((fx, "XAU_USD"))
        pairs.append((fx, "USOIL"))
    pairs.append(("XAU_USD", "USOIL"))
    pairs.append(("DXY", "XAU_USD"))
    pairs.append(("DXY", "USOIL"))
    return tuple(pairs)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def _rolling_best_lag(
    left: pd.Series,
    right: pd.Series,
    *,
    window: int,
    lags: tuple[int, ...],
) -> tuple[pd.Series, pd.Series]:
    corr_by_lag = {lag: left.rolling(window).corr(right.shift(lag)) for lag in lags}
    corr_frame = pd.DataFrame(corr_by_lag, index=left.index, dtype=float)
    valid_mask = corr_frame.notna().any(axis=1)
    abs_corr = corr_frame.abs().fillna(-np.inf)
    best_lag = abs_corr.idxmax(axis=1).astype(float)
    best_lag = best_lag.where(valid_mask, np.nan)
    best_score = pd.Series(np.nan, index=left.index, dtype=float)
    for lag in lags:
        best_score = best_score.where(best_lag != float(lag), corr_frame[lag])
    return best_score, best_lag


def build_global_market_context(
    processed_frames: Mapping[str, pd.DataFrame],
    *,
    timeframe: str,
) -> pd.DataFrame:
    if not _is_supported_timeframe(timeframe):
        raise ValueError(
            f"Cross-asset context only supports {sorted(SUPPORTED_CROSS_ASSET_TIMEFRAMES)}"
        )

    prepared: list[pd.DataFrame] = []
    available_symbols: list[str] = []
    for instrument in CONTEXT_SYMBOLS:
        frame = processed_frames.get(instrument)
        if frame is None or frame.empty or "timestamp" not in frame.columns:
            continue
        prepared.append(
            _prepare_close_frame(frame, instrument=instrument, timeframe=timeframe)
        )
        available_symbols.append(instrument)

    if not prepared:
        return pd.DataFrame(columns=["timestamp", "timeframe"])

    context = prepared[0]
    for frame in prepared[1:]:
        context = context.merge(frame, on="timestamp", how="outer")
    context = context.sort_values("timestamp").reset_index(drop=True)
    context["timeframe"] = timeframe

    for instrument in available_symbols:
        price = pd.to_numeric(context[instrument], errors="coerce").astype(float)
        context[f"{instrument}__logret"] = np.log(price).diff()

    for left, right in _iter_correlation_pairs():
        if left not in available_symbols or right not in available_symbols:
            continue
        left_ret = context[f"{left}__logret"]
        right_ret = context[f"{right}__logret"]
        for window in HORIZONS_BY_TIMEFRAME[timeframe]:
            corr_col = f"corr_{left}__{right}__w{window}"
            z_col = f"corr_z_{left}__{right}__w{window}"
            corr = left_ret.rolling(window).corr(right_ret)
            context[corr_col] = corr
            context[z_col] = _rolling_zscore(corr, window)

    for left, right in LAG_SCAN_PAIRS:
        if left not in available_symbols or right not in available_symbols:
            continue
        left_ret = context[f"{left}__logret"]
        right_ret = context[f"{right}__logret"]
        for window in HORIZONS_BY_TIMEFRAME[timeframe]:
            best_score, best_lag = _rolling_best_lag(
                left_ret,
                right_ret,
                window=window,
                lags=LAG_SCAN_LAGS[timeframe],
            )
            score_col = f"lagcorr_best_{left}__{right}__w{window}"
            lag_col = f"lagcorr_best_lag_{left}__{right}__w{window}"
            context[score_col] = best_score
            context[lag_col] = best_lag

    keep_columns = [
        "timestamp",
        "timeframe",
        *[
            column
            for column in context.columns
            if column.startswith("corr_")
            or column.startswith("corr_z_")
            or column.startswith("lagcorr_best_")
        ],
    ]
    return context[keep_columns].copy()


def load_raw_context_frames(
    *,
    raw_data_root: str | Path,
    timeframe: str,
    instruments: tuple[str, ...] = CONTEXT_SYMBOLS,
) -> dict[str, pd.DataFrame]:
    root = Path(raw_data_root)
    out: dict[str, pd.DataFrame] = {}
    for instrument in instruments:
        path = root / f"{instrument}_{timeframe}.parquet"
        if not path.exists():
            continue
        frame = normalize_candle_schema(pd.read_parquet(path), require_volume=True)
        out[instrument] = frame.reset_index(drop=True)
    return out


def build_processed_context_frames(
    *,
    primary_raw: pd.DataFrame,
    instrument: str,
    timeframe: str,
    peer_raw_frames: Mapping[str, pd.DataFrame] | None,
    raw_data_root: str | Path | None,
    frame_builder: Callable[[pd.DataFrame, str], pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    universe = dict(peer_raw_frames or {})
    if raw_data_root is not None:
        loaded = load_raw_context_frames(
            raw_data_root=raw_data_root, timeframe=timeframe
        )
        for key, value in loaded.items():
            universe.setdefault(key, value)
    universe[instrument] = primary_raw

    processed: dict[str, pd.DataFrame] = {}
    for symbol, raw in universe.items():
        if raw is None or raw.empty:
            continue
        processed[symbol] = frame_builder(raw.copy(), symbol)
    return processed


def resolve_cross_asset_inputs(
    primary_raw: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
    market_context: pd.DataFrame | None = None,
    processed_frames: Mapping[str, pd.DataFrame] | None = None,
    peer_raw_frames: Mapping[str, pd.DataFrame] | None = None,
    raw_data_root: str | Path | None = "data/raw",
    partner_builder: Callable[[pd.DataFrame, str], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame | None, dict[str, pd.DataFrame]]:
    if not _is_supported_timeframe(timeframe):
        raise ValueError(
            f"Cross-asset context only supports {sorted(SUPPORTED_CROSS_ASSET_TIMEFRAMES)}"
        )

    resolved_market_context = market_context
    if resolved_market_context is None:
        raw_context_frames: dict[str, pd.DataFrame] = {}
        for symbol, frame in (peer_raw_frames or {}).items():
            if frame is None or frame.empty:
                continue
            raw_context_frames[symbol] = normalize_candle_schema(
                frame,
                require_volume=True,
            )
        if raw_data_root is not None:
            loaded = load_raw_context_frames(
                raw_data_root=raw_data_root,
                timeframe=timeframe,
            )
            for symbol, frame in loaded.items():
                raw_context_frames.setdefault(symbol, frame)
        if instrument in CONTEXT_SYMBOLS:
            raw_context_frames[instrument] = normalize_candle_schema(
                primary_raw,
                require_volume=True,
            )
        resolved_market_context = build_global_market_context(
            raw_context_frames,
            timeframe=timeframe,
        )

    resolved_processed = dict(processed_frames or {})
    missing_partners = [
        partner
        for partner, _ in SMT_PARTNERS.get(instrument, ())
        if partner not in resolved_processed
    ]
    if missing_partners and partner_builder is not None:
        raw_partner_frames: dict[str, pd.DataFrame] = {}
        for symbol, frame in (peer_raw_frames or {}).items():
            if frame is None or frame.empty:
                continue
            raw_partner_frames[symbol] = normalize_candle_schema(
                frame,
                require_volume=True,
            )
        if raw_data_root is not None:
            loaded = load_raw_context_frames(
                raw_data_root=raw_data_root,
                timeframe=timeframe,
                instruments=tuple(missing_partners),
            )
            for symbol, frame in loaded.items():
                raw_partner_frames.setdefault(symbol, frame)
        for partner in missing_partners:
            raw_partner = raw_partner_frames.get(partner)
            if raw_partner is None or raw_partner.empty:
                continue
            resolved_processed[partner] = partner_builder(raw_partner.copy(), partner)

    return resolved_market_context, resolved_processed


def _pair_column_candidates(
    prefix: str, a: str, b: str, window: int
) -> tuple[str, str]:
    return (
        f"{prefix}_{a}__{b}__w{window}",
        f"{prefix}_{b}__{a}__w{window}",
    )


def _attach_named_columns(
    out: pd.DataFrame,
    *,
    market_context: pd.DataFrame,
    instrument: str,
    timeframe: str,
) -> pd.DataFrame:
    working = add_alignment_key(out, instrument=instrument, timeframe=timeframe)
    context = market_context.copy()
    context["timestamp"] = pd.to_datetime(
        context["timestamp"], utc=True, errors="coerce"
    )
    merged = working.merge(
        context,
        left_on=ALIGNMENT_KEY,
        right_on="timestamp",
        how="left",
        suffixes=("", "__context"),
    )
    merged = merged.drop(columns=[ALIGNMENT_KEY, "timestamp__context"], errors="ignore")

    if instrument in FX_SYMBOLS:
        partners = [symbol for symbol in FX_SYMBOLS if symbol != instrument] + [
            "XAU_USD",
            "USOIL",
        ]
        lag_pairs: list[tuple[str, str]] = []
    elif instrument == "XAU_USD":
        partners = list(FX_SYMBOLS) + ["USOIL", "DXY"]
        lag_pairs = [("XAU_USD", "USOIL")]
    elif instrument == "USOIL":
        partners = list(FX_SYMBOLS) + ["XAU_USD", "DXY"]
        lag_pairs = [("XAU_USD", "USOIL"), ("USOIL", "DXY")]
    elif instrument == "DXY":
        partners = ["XAU_USD", "USOIL"]
        lag_pairs = [("USOIL", "DXY")]
    else:
        partners = []
        lag_pairs = []

    for partner in partners:
        partner_token = _partner_token(partner)
        for window in HORIZONS_BY_TIMEFRAME[timeframe]:
            for candidate in _pair_column_candidates(
                "corr", instrument, partner, window
            ):
                if candidate in merged.columns:
                    merged[f"xasset_corr_{partner_token}_w{window}"] = merged[candidate]
                    break
            for candidate in _pair_column_candidates(
                "corr_z", instrument, partner, window
            ):
                if candidate in merged.columns:
                    merged[f"xasset_corr_z_{partner_token}_w{window}"] = merged[
                        candidate
                    ]
                    break

    for left, right in lag_pairs:
        pair_alias = _pair_alias(left, right)
        for window in HORIZONS_BY_TIMEFRAME[timeframe]:
            score_col = f"lagcorr_best_{left}__{right}__w{window}"
            lag_col = f"lagcorr_best_lag_{left}__{right}__w{window}"
            if score_col in merged.columns:
                merged[f"xasset_lagcorr_{pair_alias}_best_w{window}"] = merged[
                    score_col
                ]
            if lag_col in merged.columns:
                merged[f"xasset_lagcorr_{pair_alias}_best_lag_w{window}"] = merged[
                    lag_col
                ]

    drop_candidates = [
        column
        for column in merged.columns
        if column.startswith("corr_")
        or column.startswith("corr_z_")
        or column.startswith("lagcorr_best_")
        or column.endswith("__logret")
        or column in CONTEXT_SYMBOLS
        or column == "timeframe"
    ]
    return merged.drop(columns=drop_candidates, errors="ignore")


def _align_partner_for_primary(
    primary_frame: pd.DataFrame,
    partner_frame: pd.DataFrame,
    *,
    primary_instrument: str,
    partner_instrument: str,
    timeframe: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = add_alignment_key(
        primary_frame,
        instrument=primary_instrument,
        timeframe=timeframe,
    )
    right = add_alignment_key(
        partner_frame,
        instrument=partner_instrument,
        timeframe=timeframe,
    )
    partner_columns = [
        "timestamp",
        ALIGNMENT_KEY,
        *[
            column
            for column in partner_frame.columns
            if column in {"close", "atr_14", *SMT_REQUIRED_COLUMNS_FOR_ATTACH()}
            or column.startswith("swing_")
        ],
    ]
    partner_aligned = right[partner_columns].copy()
    merged = left[["timestamp", ALIGNMENT_KEY]].merge(
        partner_aligned,
        on=ALIGNMENT_KEY,
        how="left",
        suffixes=("", "__partner"),
    )
    partner_out = merged.drop(columns=[ALIGNMENT_KEY]).copy()
    primary_out = left.drop(columns=[ALIGNMENT_KEY]).copy()
    return primary_out, partner_out


def SMT_REQUIRED_COLUMNS_FOR_ATTACH() -> tuple[str, ...]:
    return (
        "swing_high_confirm_flag",
        "swing_low_confirm_flag",
        "swing_high_confirm_origin_idx",
        "swing_low_confirm_origin_idx",
        "swing_high_confirm_price",
        "swing_low_confirm_price",
    )


def attach_cross_asset_context(
    frame: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
    market_context: pd.DataFrame | None,
    processed_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    if not _is_supported_timeframe(timeframe):
        return out

    if market_context is not None and not market_context.empty:
        out = _attach_named_columns(
            out,
            market_context=market_context,
            instrument=instrument,
            timeframe=timeframe,
        )

    partner_specs = SMT_PARTNERS.get(instrument, ())
    best_partner = np.full(len(out), None, dtype=object)
    best_score = np.full(len(out), np.nan, dtype=float)
    any_flag = np.zeros(len(out), dtype=np.int8)

    for partner, relation_sign in partner_specs:
        if processed_frames is None or partner not in processed_frames:
            continue
        primary_aligned, partner_aligned = _align_partner_for_primary(
            out,
            processed_frames[partner],
            primary_instrument=instrument,
            partner_instrument=partner,
            timeframe=timeframe,
        )
        smt_frame = add_smt_divergence(
            primary_aligned,
            partner_aligned,
            inverse=relation_sign == -1,
            partner_name=_partner_token(partner),
        )
        keep = [
            f"xasset_smt_{_partner_token(partner)}_bull_flag",
            f"xasset_smt_{_partner_token(partner)}_bear_flag",
            f"xasset_smt_{_partner_token(partner)}_dir",
            f"xasset_smt_{_partner_token(partner)}_score",
            f"xasset_smt_{_partner_token(partner)}_expected_relation",
        ]
        for column in keep:
            out[column] = smt_frame[column].to_numpy()
        score_col = f"xasset_smt_{_partner_token(partner)}_score"
        dir_col = f"xasset_smt_{_partner_token(partner)}_dir"
        partner_score = pd.to_numeric(out[score_col], errors="coerce").to_numpy(
            dtype=float
        )
        partner_dir = (
            pd.to_numeric(out[dir_col], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.int8)
        )
        partner_active = np.isfinite(partner_score) & (partner_dir != 0)
        replace_mask = partner_active & (
            ~np.isfinite(best_score) | (partner_score > best_score)
        )
        best_score = np.where(replace_mask, partner_score, best_score)
        best_partner = np.where(replace_mask, partner, best_partner)
        any_flag = np.where(partner_active, 1, any_flag).astype(np.int8)

    out["xasset_smt_any_flag"] = any_flag
    out["xasset_smt_best_partner"] = pd.Series(best_partner, dtype="object").where(
        pd.Series(any_flag == 1),
        None,
    )
    out["xasset_smt_best_score"] = pd.Series(best_score).where(any_flag == 1, np.nan)
    return out


def persist_market_context(
    market_context: pd.DataFrame,
    *,
    features_root: str | Path,
    timeframe: str,
    variant: str,
    frontier_from_ts: pd.Timestamp | None,
    full_rebuild: bool,
) -> list:
    if market_context.empty:
        return []
    dataset = f"market_context_{variant}"
    return persist_partitioned_dataset(
        market_context,
        base_dir=features_root,
        dataset=dataset,
        symbol=GLOBAL_CONTEXT_SYMBOL,
        timeframe=timeframe,
        frontier_from_ts=frontier_from_ts,
        full_rebuild=full_rebuild,
    )
