# Changelog

## [0.0.2] — 2026-06-06

### Changed
- Notes (`format="md"`) reflow now batches transcript windows through a single
  generation pass and stops at the model's turn boundary instead of running to
  the token cap. ~10× faster Gemma reflow on CUDA with byte-identical output.

## [0.0.1] — 2026-06-05

Initial release.
