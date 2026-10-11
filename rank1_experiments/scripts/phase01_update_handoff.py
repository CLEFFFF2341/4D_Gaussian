"""Replace the initial handoff with the verified phase-01 continuation inputs."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for name in ['README.md','RESUME_PROMPT.txt']:
    backup=ROOT/'manifests'/('initial_'+name.replace('.md','.txt'))
    if not backup.exists():
        backup.write_bytes((ROOT/name).read_bytes())
(ROOT/'README.md').write_text('''# MoRGS research workspace

Phase 01: PASS for the environment/source prerequisite gate. Full baseline reproduction remains pending. Phases 02 and 03 have not started.

- Latest Chinese report: reports/phase01_environment_recovery.md.
- Executed command reference: reports/phase01_recovery_commands.md.
- Initial source audit, retained as history: reports/phase01_source_audit.md.
- Current machine-readable state: experiment_state.json.
- Official core environment: envs/morgs-cu121 (Python 3.10.18, torch 2.5.1, CUDA 12.1).
- Working copy: working/MoRGS; two source corrections and a Windows SAM2 CUDA binary are recorded.
- Exact dependencies and weights: manifests/conda_package_lock.json, pip_official_freeze.txt, native_wheel_lock.json, weight_lock.json.
- Activation/execution: scripts/run_official.cmd. Use absolute script paths.
- Streaming evaluation adapter: scripts/evaluate_streaming.py; checked against upstream metrics on synthetic images.
- Continuation: RESUME_PROMPT.txt.

All required native kernels, pretrained teachers, a two-frame GaussianModel optimizer smoke and the evaluation contract passed. Raw data, the Ex4DGS environment and pristine source are preserved. Tiny diagnostics do not establish full-scene model quality or resource use. The installed motion rasterizer contains the mask-stride correction; this is an engineering variant, not an unmodified author benchmark.

Proceed with phase 02: isolated dynerf data preparation, frozen splits, train-only initialization, camera reprojection and teacher-cache checks. Do not rerun the historical phase01_finalize.py; current evidence is frozen by phase01_recovery_finalize.py.
''',encoding='utf-8')
(ROOT/'RESUME_PROMPT.txt').write_text('''Continue in E:\\4D_Gaussian\\rank1_experiments.
Read experiment_state.json and reports/phase01_environment_recovery.md. Then read the original plan and common constraints in E:\\4D_Gaussian\\rank1_experiment_plan_20261010.
Phase 01 prerequisites passed. Do not repeat the initial failed environment installation or use the older diagnostic venv for new experiments.
Use envs/morgs-cu121: Python 3.10.18, torch 2.5.1, torchvision 0.20.1, CUDA runtime/nvcc 12.1. Use scripts/run_official.cmd to activate DLL and cache paths. Preserve the Ex4DGS environment, original data and pristine source. No driver installation/reset/reboot.
The primary source is pinned to 5c62ae88b8fb2e3d4213329dd1ec3a61c66f8902. Original 3290 files are verified. The active working tree is working/MoRGS. Original source changes are only the 8-column CUDA mask stride and Windows separator normalization. Native wheel and SAM2 binary hashes are recorded. Keep author B0 and corrected engineering variants separate in future comparisons.
All native operators passed in the target environment: simple-knn, torch_cluster, both rasterizers, SAM2 connected components and torchac. Actual GaussianModel initialization and one optimizer step in each of two synthetic frames passed. Training and metrics CLI entries load. This is not full scene training, a densification test or a resume test.
SEA-RAFT weights: weights/Tartan-C-T-TSKH-spring540x960-M.pth, converted from the pinned official HF safetensors by restoring only shared-module aliases. Strict load and tensor equality passed. Config: working/MoRGS/RAFT/config/eval/spring-M.json.
SAM2 checkpoint: weights/sam2.1_hiera_large.pt. Configuration: configs/sam2.1/sam2.1_hiera_l.yaml. Keep the vendored init_state3 interface. Do not replace it with a new official Python package. Both pretrained helpers ran on coffee_martini/cam02 frames 0/1 at 676x507. Teacher checkpoints are hash-locked; no formal cache exists yet.
Next execute phase 02. N3DV all six raw scenes and poses are in E:\\4DGS\\data, with 120 cameras. Build separate data/dynerf/<scene> directories containing camera directories only, root-level zero-padded PNGs and audited points3D_downsample2.ply. Validate train-only initialization, projection and dimensions. cam00 is test; proposed dev validation is cam01 in the two development scenes. Splits are not frozen. Remap flow_views from fixed original camera names after removing validation; runtime flow_views are internal training indices. The earlier mismatch claim was withdrawn.
Use the pinned FFmpeg 7.1 binary and preserve per-frame extraction logs. cut_roasted_beef/cam10 had a transient first-sweep decode error but matched its ZIP and subsequently decoded four times. Do not silently remove or replace original cameras/files. Other four scenes lack complete sequential decode checks.
Check teacher input normalization and resizing against the vendored author behavior before caching. SAM init_state3 preprocessing comments mention normalization but its actual return does not apply mean/std. Preserve author behavior for B0 and record shared-protocol changes explicitly. Verify online/cache equality, forward/backward direction, masks and FP16 error using only training views.
Meet Room is not downloaded; official mirrors are recorded, exact scenes/test camera unresolved. The second backbone is hustvl/4DGaussians, not Ex4DGS at E:\\4DGS.
The flow projection diagnostic remains near 2x in its known-translation example; current-geometry grid and residual excluding RGB offset still need common-control work. Do not attribute these engineering fixes to the research likelihood. Persistent UID and full checkpoint/resume are pending later-phase implementation.
The source has no render.py. Use the verified training_report image contract, and scripts/evaluate_streaming.py for per-frame evaluation with the same vendored metrics. Its synthetic outputs match upstream per-frame metrics exactly; actual sequence and resume behavior are not yet verified. All smoke metrics are fixtures, never baseline quality results.
Current finalizer: scripts/phase01_recovery_finalize.py. The historical scripts/phase01_finalize.py is retired for current state. Update evidence and state as phase 02 advances, without claiming later phases are complete.
''',encoding='utf-8')
print('Updated handoff to the verified phase-01 environment; phase 02 is pending.')
