# Executed phase-01 recovery commands

All paths below refer to this workspace. These commands were executed; none is a full scene training command. Logs retain the failed initial attempts as well as successful retries.

The official environment was created from local packages after recovering the missing CUDA archive and checking its official SHA256. The original URL lock remains separate from the local installation list.

```powershell
& 'C:\Users\CLEFFFF\anaconda3\Scripts\conda.exe' create --prefix 'E:\4D_Gaussian\rank1_experiments\envs\morgs-cu121' --file 'E:\4D_Gaussian\rank1_experiments\manifests\conda_explicit_local.txt' --offline --yes --json
& 'E:\4D_Gaussian\rank1_experiments\envs\morgs-cu121\python.exe' -m pip install -r 'E:\4D_Gaussian\rank1_experiments\manifests\pip_recovery_requirements.txt'
```

Before reusing the install command, consult pip_official_freeze.txt for the exact transitive versions. The creation command's captured JSON file was empty despite exit 0; the resulting environment, package inventory, and GPU smoke establish success.

Native builds ran through scripts/build_official_native.cmd, once each with the working source directories submodules/simple-knn, submodules/gaussian-rasterization-grad, and submodules/diff-gaussian-rasterization. scripts/build_official_sam2.cmd compiled the separately pinned SAM2 CUDA source into working/MoRGS/sam2. These wrappers activate the isolated environment, use VS2019, CUDA_HOME pointing at that environment, sm_89, and MAX_JOBS=2. The installed cp310 wheels are hash-locked in native_wheel_lock.json.

Use absolute script paths because run_official.cmd changes the current directory:

```powershell
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase01_teacher_smoke.py' convert
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase01_teacher_smoke.py' raft
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase01_teacher_smoke.py' sam
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase01_official_ops_smoke.py' knn
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase01_official_ops_smoke.py' raster
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official_codec.cmd'
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase01_native_probe.py' --variant mask_stride_8 --output-name native_probe_official_mask_stride_8.json
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase01_model_smoke.py'
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' train.py --help
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' metrics_video.py --help
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\phase01_make_eval_fixture.py'
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' metrics_video.py -m 'E:\4D_Gaussian\rank1_experiments\smoke\metrics_contract'
& 'E:\4D_Gaussian\rank1_experiments\scripts\run_official.cmd' 'E:\4D_Gaussian\rank1_experiments\scripts\evaluate_streaming.py' -m 'E:\4D_Gaussian\rank1_experiments\smoke\metrics_contract' --reference-dir 'E:\4D_Gaussian\rank1_experiments\smoke\metrics_contract'
```

The rectangle SAM prompt and synthetic metrics fixture validate interfaces only. The model smoke has two steps and no scene loader, densification, checkpoint/resume, or benchmarking claim.

Important logs: official_torch_smoke.log, official_build_*.log, teacher_convert_official.log, teacher_raft_official.log, teacher_sam_official.log, official_model_smoke.log, official_evaluation_smoke.log, streaming_evaluation_smoke.log, official_pip_check.log. Earlier conversion and fixture failures are retained separately and explained in the recovery report.

Finalization uses phase01_recovery_finalize.py. The historical phase01_finalize.py must not overwrite the recovered state.
