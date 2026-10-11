# Phase 01 workspace

Status: PARTIAL. The source audit and bounded GPU diagnostics are recorded. The full training prerequisite gate has not passed.

- Chinese audit: `reports/phase01_source_audit.md`.
- Machine-readable state: `experiment_state.json`.
- Source/dependency/data/config evidence: `manifests/`.
- Actual execution and failure records: `logs/`.
- Isolated mask-stride correction: `patches/mask_stride_8.patch`.
- Diagnostic and audit scripts: `scripts/`.
- Continuation instructions: `RESUME_PROMPT.txt`.

Original datasets, Ex4DGS source/environment, research results and checkpoints are preserved. The diagnostic native venv uses older system site packages; it is not the official MoRGS environment. No baseline or proposed-method training metrics exist yet.
