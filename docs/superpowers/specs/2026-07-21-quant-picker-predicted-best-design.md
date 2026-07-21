# Quant picker: highlight predicted-best variant per tier

## Problem

`omm install <repo>` with an ambiguous repo (multiple `.gguf` files) shows
`_pick_quant_variant` in `src/omm/cli.py`. Today every variant only gets a
plain "fits / may not fit / fit unknown" note from the memory-size heuristic
in `hub.rank_quant_variants`. With repos that expose 15-25 quant files, the
user can't tell which of the several "fits" variants is actually fastest —
they all look equally viable.

`omm search` already runs candidates through the ML ensemble predictor
(`predictor.predict_speed`) and marks repo-level results red when predicted
unviable. The quant picker has no equivalent positive signal from that same
model.

## Goal

Reuse the existing prediction model to mark the predicted-fastest variant
**within each quant-bits tier** (e.g. best among all ~Q4 variants, best among
all ~Q5 variants, etc.) in green, so the user can quickly spot the best pick
at whatever quality/size level they want.

## Design

### Where

`_pick_quant_variant` in `src/omm/cli.py` (around line 718), after
`rank_quant_variants` has produced the sorted `variants` list and before
`choices` are built.

### Steps

1. Load the cached predictor model: `predictor.load_cached_model()`.
   - If `None` (no cached model), skip prediction entirely — render exactly
     as today (heuristic-only fits/may-not-fit/unknown notes, no color).
     This mirrors the existing fallback pattern in `omm search`, where
     `trees is None` means no red-marking either.
2. If a model is present, for every variant with `fits is True`:
   - Build a candidate dict `{"repo_id": error.repo_id, "filename": v.filename}`.
   - Call `predictor.predict_speed(trees, hw, candidate)`.
   - If the predictor can't resolve a param count from the filename/repo_id
     (returns `<= 0` for a reason other than "doesn't fit" — i.e. the
     parsers in `featurize.py` found nothing), treat as "unknown speed" and
     exclude it from tier-best consideration. Do not force a green mark on
     a guess.
3. Group the fits-True, speed-resolved variants by `quant_bits` (exact float
   value already on `QuantVariant`, e.g. 4.0, 5.0, 6.0, 8.0 — this already
   separates Q4 from Q5 etc. the same way the existing sort does).
4. Within each group, pick the single variant with the highest predicted
   speed. That variant is "predicted fastest" for its tier.
5. Ties (identical predicted speed within a tier — common, since speed
   depends only on quant_bits/param_count/model_size, not on filename
   details like `Q4_K_M` vs `Q4_0`): keep the first one in the existing
   sort order (already fits-desc, quant_bits-desc, so this is a stable,
   deterministic pick — no new tie-break logic needed).

### Rendering

Existing loop building `choices` changes only for tier-best entries:

```python
if v.filename in tier_best_filenames:
    title = [("fg:green bold", f"{v.filename}  ({note}, predicted fastest)")]
else:
    title = f"{v.filename}  ({note})"
choices.append(questionary.Choice(title=title, value=v.filename))
```

`questionary.Choice.title` already accepts a list of `(style, text)` tuples
(prompt_toolkit formatted text), so no new dependency.

Sort order of `choices` is unchanged — green is a color overlay on the
existing fits-desc/quant_bits-desc order, not a re-rank.

### Non-goals

- Not changing `rank_quant_variants`' sort order or the heuristic fit
  calculation.
- Not adding a new fallback param-count source for the predictor (accepting
  that a handful of filenames without parseable param counts just won't get
  a green mark, same as they already show "fit unknown" today).
- Not touching `omm search`'s existing red-marking behavior.

## Testing

- Unit test on the tier-grouping/best-pick logic with a synthetic
  `list[QuantVariant]` + stub `predictor.predict_speed` returning distinct
  speeds per tier, asserting exactly one green pick per tier and that ties
  resolve to the existing sort order's first element.
- Unit test confirming no crash / no color when `load_cached_model()`
  returns `None`.
