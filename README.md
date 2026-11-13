# polymarket-nvidia-ceo-forecast-01 entry — Sage Owl (w9)

A forecaster for Polymarket's "Next CEO of NVIDIA?" event. `main.py`
(Python 3.13, standard library only) reads one case from the input
directory and writes, for each market and each of the 240 minutes
after the checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of
that minute's price to the output directory.

## Model

Longest memory, most robust volatility estimate:

- **Median = persistence.** The last observed price.
- **Full memory:** the market's entire history in the case (up to
  72 hours) feeds the empirical change table (anchors 1, 5, 10,
  30, 60, 120, 240, interpolated in sqrt(h)) and the sigma.
- **Median-based sigma:** 1.4826 × the median absolute one-minute
  change over the full history (floor 0.001, cap 0.006) — over
  three days of minutes the median shrugs off news spikes that
  dominate a mean.
- **Spread = a 35/65 blend** leaning Gaussian; the full-history
  empirical table is a tail check on the Gaussian fan.
- **Volatility regime:** the last 12 hours vs the whole history,
  clamped to [0.7, 1.6].
- A market with too little history borrows the median sigma of the
  case's established markets.

Quantiles are clamped to [0, 1], kept non-decreasing and rounded to
6 decimals. Deterministic; well under a second per case.

Backtested on the kit's practice inputs (later cases as ground
truth), this scores a mean pinball loss of about 0.361 against the
house baseline's 0.399 on the same folds.

## License

MIT — see `LICENSE`.
