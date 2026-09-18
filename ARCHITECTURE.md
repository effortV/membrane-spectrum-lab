# Architecture and scientific gates

```text
NF.xlsx / RO.xlsx ─┐
NF-N/O, RO-N/O ───┼─> versioned audit + canonical provenance ─┐
local/API PDFs ───┴─> page text + selected visual evidence ───┤
                                                            v
             DeepSeek hypothesis agent -> descriptor DSL -> applicability gate
                                                            |
                                                            v
         nested DOI-group CV <-> counterexamples <-> literature novelty search
                                                            |
                                                            v
               reject / revise / tentative candidate / external validation
```

## Why this is an agent rather than a one-shot model

The main model has a bounded tool loop. It can inspect the actual data profile, search page-located evidence, inspect hypotheses and validation runs, and add OpenAlex discoveries to the acquisition queue. Expensive or state-changing stages remain explicit CLI commands, so the model cannot silently download hundreds of papers or launch costly VLM runs.

## Evidence levels

1. **E0 — proposed:** schema-valid idea with explicit observables and falsification plan.
2. **E1 — calculable:** inputs exist, units are resolved, and extraction/fit uncertainty is recorded.
3. **E2 — internally supported:** added value survives identical nested grouped folds and sensitivity tests.
4. **E3 — literature triangulated:** direct evidence and counterexamples are page-located; novelty search completed.
5. **E4 — externally validated:** independent DOI/time/lab holdout or new experiment confirms the conditional effect.

Only E4 can be described as a validated discovery. Earlier levels are candidates.

## Leakage and overclaim prevention

- The target test fold is never used to choose a split, feature, model, or hyperparameter.
- Both NF and RO rows from the same paper share one group.
- Missing targets are excluded only because they cannot be scored; no target-IQR filtering is applied before splitting.
- Baseline and augmented models use identical outer folds. Hyperparameters are selected only inside the training groups.
- Negative R² and failed candidates are retained in the run ledger.
- Every run saves input hash, split hash, fold predictions and metrics.

## v0.2 operational gates

- Candidate symbols cannot overwrite source columns; input roles/citation IDs are verified in both proposal and materialization. A bad candidate is rejected individually. Optional dataset/input-bound gates are executable; prose applicability is not chemical validation.
- Energy-grid quadrature and a fixed reference grid separate spectrum shape from point-density artifacts. Different energy windows, baseline/charging and digitization errors remain unresolved confounders.
- Default evaluation uses resolved DOI and observed candidate inputs only, with cohort counts. Target/group leakage, duplicated predictors and split DOI are rejected in the backend, not just in the UI.
- Fold sign flips are exploratory; overlapping training folds and adaptive/multiple candidate selection invalidate discovery-significance claims. Group-balanced OOF errors and a training-mean dummy benchmark accompany scores.
- Append-only LLM network events distinguish actual successful/failed requests from cache hits; hard per-action request budgets do not estimate monetary spend. Retrying an ambiguous timeout can still duplicate provider billing.
- Runtime health reports inspect stale manifests, source hashes, citation/schema validity, failed documents and missing DOI without calling any API. The research agent can inspect those reports, variable metadata, full validation summaries, and bounded novelty-search metadata.
- Local UI startup needs no administrator. Optional hashed-password access is suitable for a trusted/local research deployment, not Internet exposure; it does not provide transport encryption.
