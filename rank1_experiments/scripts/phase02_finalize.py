"""Finalize Phase 02 only after all executed prerequisite evidence is present."""
import ast
from datetime import datetime,timezone,timedelta
import difflib
import json
from pathlib import Path
import shutil
import numpy as np
from phase02_prepare_data import ROOT,DEV,FLOW,sha,save
def load(name):return json.loads((ROOT/'manifests'/name).read_text(encoding='utf-8'))
def run():
    data=load('dataset_manifest.json');teachers=load('teacher_manifest.json');verify=load('teacher_cache_verification.json')
    timestamps=load('video_timestamp_audit.json');loader=load('phase02_loader_checks.json');numeric=load('phase02_numerical_checks.json');meet=load('meetroom_manifest.json')
    assert all(x['status']=='PASS' for x in [data,teachers,verify,timestamps,loader,numeric])
    assert len(data['records'])==38 and len(timestamps['records'])==120
    count=0;total_bytes=0
    for record_path in data['records']:
        row=json.loads(Path(record_path).read_text(encoding='utf-8'))
        assert row['status']=='PASS' and sha(row['video'])==row['video_sha256']
        for protocol,n in [('dev',90),('final',300)]:
            assert len(row['frames'][protocol])==n
            for i,frame in enumerate(row['frames'][protocol]):
                assert frame['frame']==i and frame['source_video_frame']==i
                assert sha(frame['path'])==frame['sha256'],frame['path']
                total_bytes+=frame['bytes'];count+=1
    assert count==14820
    geometries=[load(f'geometry_{protocol}_{scene}.json') for protocol in ['dev','final'] for scene in DEV]
    for g in geometries:assert g['status']=='PASS' and sha(g['ply_path'])==g['ply_sha256']
    # Update the extraction convention, without changing frozen camera/frame splits.
    for protocol in ['dev','final']:
        p=ROOT/f'manifests/splits_{protocol}.json';m=json.loads(p.read_text(encoding='utf-8'))
        m['resolution_convention']['resize']='FFmpeg area raw->formal x0.5; OpenCV INTER_AREA formal->dev x0.5; no crop or rotation'
        save(p,m)
    work=ROOT/'working/MoRGS';pristine=Path(load('source_lock.json')['source_path'])
    changed=['scene/dataset_readers.py','utils/loader_utils.py']
    patch=''
    entries=[]
    for name in changed:
        before=(pristine/name).read_text(encoding='utf-8');after=(work/name).read_text(encoding='utf-8')
        patch+=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        entries.append(dict(path=name,original_sha256=sha(pristine/name),working_sha256=sha(work/name)))
    patch_path=ROOT/'manifests/phase02_data_isolation.patch';patch_path.write_text(patch,encoding='utf-8')
    save(ROOT/'manifests/phase02_source_patch.json',dict(status='PASS',patch=str(patch_path),patch_sha256=sha(patch_path),files=entries,
       purpose='Read dimensions from metadata, enforce train/test roots and directory ordering; default author roots remain supported.',
       research_loss_changes=False,phase01_patches_retained=True))
    script_names=['phase02_prepare_data.py','phase02_teacher_cache.py','phase02_geometry.py','phase02_protocol_checks.py',
       'phase02_meetroom_manifest.py','phase02_video_timestamps.py','protocol_cache.py','phase02_cache_verify.py','phase02_finalize.py']
    for name in script_names:ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8'))
    save(ROOT/'manifests/phase02_artifact_lock.json',dict(scripts={n:sha(ROOT/'scripts'/n) for n in script_names},
       source_patches=entries,development_pngs=3420,formal_pngs=11400,all_png_hashes_verified=True,
       extracted_bytes=total_bytes,flow_npz=1440,sam_npz=88))
    now=datetime.now(timezone(timedelta(hours=8))).isoformat()
    lines=[f'# 阶段 02：数据、划分与教师缓存审计\n\n更新时间：{now}。状态：**通过开发数据门槛**。完整 MoRGS 论文复现尚未完成，未运行场景训练。\n',
      '## 实际完成范围\n\n复用 E:\\4DGS\\data 中的原始发布视频，不修改旧实验。coffee_martini 18 路、cut_roasted_beef 20 路均完整提取前 300 帧（1352×1014），开发片段为 0—89（676×507）。共 14,820 张 PNG；每张已完整解码并保存 SHA256，最终再次核对全部文件哈希。六场景 120 路视频均核对全部容器 PTS，0—299 使用一致的 30 FPS 时间网格。其余四场景已下载的原视频列入清单，尚未做完整顺序 RGB 解码和图像导出。\n',
      '同步证据来自容器 PTS 与 [N3DV 官方发布说明](https://github.com/facebookresearch/Neural_3D_Video)，不是独立物理同步测量。公开发布没有可复核的完整视频官方 SHA；本地哈希用于锁定字节，前阶段已对两个开发场景的 ZIP 作 CRC 检查。\n',
      '## 冻结划分与读入隔离\n\ncam00 为最终测试；开发验证为 cam01，其余真实可用视角训练。正式训练恢复 cam01。poses_bounds.npy 的原始行对应实际 MP4 排序，不能用相机编号直接当行号。派生目录中的 poses_bounds.npy 是该角色相机按原顺序截取的子集。公开全相机标定允许使用；初始化与教师只读取训练 RGB，第 0 帧点云也不使用未来 RGB。训练目录没有测试或开发验证图像。\n\n| 场景 | 开发训练视角 | 正式训练视角 | 光流原相机 | 开发运行时索引 |\n|---|---:|---:|---|---|']
    splits=load('splits_dev.json');final=load('splits_final.json')
    for scene in DEV:
        s=splits['scenes'][scene];lines.append(f"| {scene} | {len(s['train'])} | {len(final['scenes'][scene]['train'])} | {', '.join(s['flow_cameras'])} | {s['flow_runtime_indices']} |")
    lines += ['\n实际 loader 与 scene reader 已用图像文件打开追踪验证四个训练目录只读其训练 RGB；错误使用默认 test_indices=[0] 或把测试根目录传给训练 loader 会明确拒绝。该补丁只处理读入，不修改研究损失。**原 train.py 仍把测试数据和训练循环绑定，阶段 03 必须接入独立验证/最终评估与缓存 adapter；不能直接把旧命令当作严格隔离训练入口。**\n',
      '## 相机、缩放和初始化\n\nLLFF down/right/back 转成 CV right/down/front，w2c=Rᵀ、t=−RᵀC。原图 2704×2028，无裁剪；FFmpeg area 缩小 2 倍得到正式图像，再由 OpenCV INTER_AREA 缩小 2 倍得到开发图像。像素中心用 (x+0.5)s−0.5；流的 x/y 分量分别乘对应尺寸比例。保持 B0 的对称 FoV 光心 (W−1)/2；它和严格原 K 的 half-pixel 缩放在开发尺寸相差 0.125 像素、正式相差 0.25 像素，分别记录，未悄悄改作者投影。poses 不含畸变系数，沿用发布图像处理，不声称重新标定或去畸变。\n',
      '**初始化协议差异：** 新点云由公开已标定相机和训练第 0 帧的 SIFT 对应三角化生成。它是可审计的稀疏开发初始化，尚不是作者链接的 COLMAP dense / downsample2 点云流程；文件同名不代表等价复现。开发和正式划分分别重建，不复用漏掉 cam01 的点云冒充正式初始化。阶段 03 必须将这套结果标为严格开发协议；与论文数字比较前补做作者初始化流程，所有方法共同使用同一初始化。\n\n| 场景/划分 | 点数 | 正确位姿极线内点比例中位数 | 错配位姿对照 |\n|---|---:|---:|---:|']
    for g in geometries:lines.append(f"| {g['scene']}/{g['protocol']} | {g['points']} | {g['correct_epipolar_median_inlier_fraction']:.3%} | {g['wrong_row_control_median_inlier_fraction']:.3%} |")
    lines += ['\n重投影使用真实训练图像对应，并检查正深度、三角化角度与各视角误差。接受点的各视角重投影误差小于 1.5 开发像素；极线对应独立于三角化筛选统计。错配位姿对照显著更差，支持实际排序和图像方向正确；这不是 3D 运动真值验证。\n',
      '## 教师与缓存\n\nSEA-RAFT M 与 SAM2.1 large 使用阶段 01 严格加载的固定权重，SHA/配置/源代码/相机/帧号/单位写入 manifest 和缓存。对两场景各 4 个真实训练视角、90 帧预计算：1,440 份方向流（89 个相邻对和额外 0→3 初始化对的前后向），88 份 SAM 刷新标签。正式 300 帧的教师缓存尚未生成，将按同一固定协议在阶段 03/正式运行前补齐。\n',
      '原版 B0 保留 FP32 未过滤光流与 heatmap、1.5 像素运动阈值和原版 SAM 跨相机列表顺序；第一个训练 step 用 0→3，之后用前帧→当前帧；SAM 在 step 1、2、10…90 刷新并保持到下一次刷新。未改 [0,1] 输入，也未修正上游注释掉的 RAFT normalization 或 SAM 自定义预处理。B0 缓存读回使用全像素有效掩码，绝不混入共享 FB 掩码。\n',
      '共享候选独立保存 FP16 流：修正上游 507→253→506→507 最后 resize 漏掉的 y 分量倍率（507/506）；输入教师不升级。共享 FB mask 由未量化 FP32 计算，在参考 1352×1014 像素单位使用 ||f+warp(b)||² ≤ 0.01(||f||²+||warp(b)||²)+0.5，并与越界和有限数值条件相交。此修正是共同工程对照，不能计入新损失的研究收益。\n',
      f"在线流抽查 {verify['online_flow_samples']} 对，B0 缓存往返最大误差 {verify['b0_cache_roundtrip_abs_max']:.8g}；在线重复最大差 {verify['online_flow_repeat_abs_max']:.8g} 像素。SAM 两场景各 4 视角重复均无变化像素。FP16 最大绝对误差 {verify['fp16_abs_max']:.8g}，全缓存平均 EPE 误差 {verify['fp16_mean_epe']:.8g} 开发像素。共享 mask 覆盖 min/mean/max 为 {verify['shared_mask_coverage_min']:.3%}/{verify['shared_mask_coverage_mean']:.3%}/{verify['shared_mask_coverage_max']:.3%}。B0 全部 720 个相机×训练 step 重放通过；这是标签/调度等价验证，训练过程端到端等价仍须阶段 03 接入后检查。B0 SAM 联合运动 mask 平均覆盖仅 {verify['b0_sam_union_coverage_mean']:.3%}，已记录而不调高覆盖；阶段 03 检查其对运动信心激活的影响，不能将数值一致性当教师运动准确性证明。\n",
      f"教师计算顺序为 RAFT 后卸载、SAM 后卸载。RAFT 进程峰值 allocated {teachers['raft_peak_allocated']/2**30:.3f} GiB，SAM {teachers['sam_peak_allocated']/2**30:.3f} GiB；总阶段 RAFT 壁钟 {teachers['raft_seconds']:.1f} 秒包含等待图像导出，不能当纯推理时间或训练预算。SAM 壁钟 {teachers['sam_seconds']:.1f} 秒。缓存重放只读取文件，没有教师模型；阶段 03 不应常驻这两个模型。\n",
      '## 导出错误与处理\n\n首次 FFmpeg 直接编码 PNG 返回成功，但 coffee/cam00 发现 3 张 IDAT 无法解码（formal 0043/0217、dev 0047）。已改为严格视频解码→原始像素管道→OpenCV PNG 编码，并逐张 Pillow 完整解码。cut/cam00 初次在 frame 190、cut/cam19 在 frame 122 被 -xerror 中止，保留日志后分别重新完整导出并通过。任何失败尝试都不计为成功，不跳帧，不修改原视频。最终通过帧数、完整解码、哈希的全部结果才进入 manifest。错误根因未独立确定，不能声称原文件有不可恢复损坏或机器故障。额外重试记录见 extraction_failures.json 和 logs/*failed*。\n',
      '## Meet Room\n\n[StreamRF 官方入口](https://github.com/AlgoHunt/StreamRF#dataset) 的 [Drive](https://drive.google.com/drive/folders/1lNmQ6_ykyKjT6UKy-SnqWoSlI5yjh3l_) 有 discussion、trimming、vrheadset，每场景 13 路 cam_0…cam_12，位姿均为 13×17 有限数组，公开标定尺寸 1280×720。官方 prepare_dataset.py 将 cam_0 改名 cam00；官方 meetroom.json 的 llffhold=100 与 loader 的 ind % hold_every 对应首视角留出；MoRGS 默认 test_indices=[0]，论文指中央视角，位姿中 row0 也都最接近相机中心。因此冻结原始测试视角 cam_0，不能盲套 N3DV 名称。使用数值后缀排序与零补齐映射，避免 cam_10 字典序错位。镜像额外 stepin 不纳入三场景协议。ModelScope tree 可达，但文件下载要求登录；通过无需登录的官方 Drive 取得小型位姿与列表，未下载 39 路视频。图像级重投影、视频帧数/FPS/完整性仍待第二数据集实际准备。\n',
      '## 产物与执行\n\n关键清单：manifests/dataset_manifest.json、splits_dev.json、splits_final.json、teacher_manifest.json、teacher_cache_verification.json、phase02_loader_checks.json、video_timestamp_audit.json、meetroom_manifest.json。每帧索引在 manifests/extraction/<scene>/<camera>.json。入口均在 scripts/，以绝对脚本路径调用 run_official.cmd。\n\n```powershell\n']
    wrapper=str(ROOT/'scripts/run_official.cmd')
    for script,args in [('phase02_prepare_data.py','extract'),('phase02_geometry.py','--protocol dev'),('phase02_geometry.py','--protocol final'),
      ('phase02_teacher_cache.py','raft'),('phase02_teacher_cache.py','sam'),('phase02_protocol_checks.py','loader'),('phase02_cache_verify.py',''),('phase02_finalize.py','')]:
        lines.append(f"& '{wrapper}' '{ROOT/'scripts'/script}' {args}")
    lines += ['```\n\n这些命令已经执行；extract/教师支持已完成单元的续跑，geometry 会重新构建对应点云。不要再次运行 prepare_data.py init 覆盖冻结划分。阶段 01 finalizer/handoff 属于历史阶段，也不要运行来覆盖阶段 02 状态。\n',
      '## 继续门槛与下一步\n\n两开发场景全部视角/帧完整、训练图像和位姿对应、缓存数值与调度抽查、开发/测试读入隔离均通过。允许进入阶段 03 的严格开发基线：先完成独立评估与缓存接入、可恢复状态，再执行 10 帧 smoke。90 帧/正式 300 帧训练尚未开始，PSNR、训练耗时与峰值显存没有结果，不能宣称方法有效或完成论文复现。\n']
    (ROOT/'reports/phase02_data_audit.md').write_text('\n'.join(lines),encoding='utf-8')
    state_path=ROOT/'experiment_state.json';state=json.loads(state_path.read_text(encoding='utf-8'))
    state.update(updated_at=now,phase='02',status='PASS',phase01='PASS',phase02='PASS',continue_gate_passed=True,
      scope='Phase 02 strict development data/cache prerequisites; author dense initialization and full baseline reproduction remain pending',
      phase02_finished_at=now,phase02_gate=dict(data_complete=True,camera_reprojection=True,teacher_cache=True,split_isolation=True),
      next_phase='03',next_action='Integrate isolated validation/evaluation and teacher replay; complete checkpoint/resume; then 10-frame strict-development B0 smoke.',
      baseline_reproduction_complete=False,training_results=None,
      phase02_protocol_differences=['Train-only sparse calibrated SIFT point cloud instead of author dense COLMAP pipeline',
        'Physically isolated dev cam01 validation and final cam00 test RGB',
        'Original B0 FP32 teacher labels separate from shared axis-resize/FB-mask FP16 labels'],
      future_validation=['End-to-end cached-vs-online B0 behavior after training adapter integration',
       'Author dense initialization pipeline before any paper-number comparison',
       'Source-grid/flow projection and offset controls','Densification identity and full checkpoint/resume',
       'Full baseline quality, training memory/duration/FPS','Low upstream B0 motion-mask coverage and confidence activation',
       'Formal 300-frame teacher caches','Meet Room video downloads/decode and image-level pose verification'])
    names=['dataset_manifest.json','splits_dev.json','splits_final.json','teacher_manifest.json','teacher_cache_verification.json',
      'phase02_loader_checks.json','phase02_numerical_checks.json','video_timestamp_audit.json','meetroom_manifest.json',
      'phase02_source_patch.json','phase02_artifact_lock.json']
    for name in names:state['manifests'][name]=sha(ROOT/'manifests'/name)
    state.setdefault('patches',{})['phase02_data_isolation.patch']=sha(patch_path)
    state['phase02_report']=str(ROOT/'reports/phase02_data_audit.md')
    save(state_path,state)
    (ROOT/'README.md').write_text('''# MoRGS Research Workspace

Phase 01 and Phase 02 prerequisites passed. No scene training or paper reproduction results yet.

Read reports/phase02_data_audit.md and experiment_state.json for current evidence and limits.
Two N3DV scenes: dev90 at 676x507, final300 at 1352x1014; train/validation/test RGB physically separate.
Teacher cache: B0 lossless FP32 vs separate shared FP16/FB-mask protocol. Author dense initialization is pending; current point clouds are train-only calibrated sparse SIFT.

Next: Phase 03 isolated training/evaluation and cache adapter, complete resume state, then 10-frame B0 smoke.
Use scripts/run_official.cmd with absolute script paths. Do not rerun historical Phase 01 finalizers or Phase 02 init.
''',encoding='utf-8')
    (ROOT/'RESUME_PROMPT.txt').write_text(f'''Continue Phase 03 in {ROOT}. Read experiment_state.json, reports/phase02_data_audit.md, and both common plan documents under E:/4D_Gaussian/rank1_experiment_plan_20261010.
Phase 01 and Phase 02 passed their stated prerequisite gates. No scene training is complete.
Use envs/morgs-cu121 and scripts/run_official.cmd, preserve E:/4DGS and all existing logs.
Two scenes have complete final300/dev90 PNGs and role-specific poses. The dev90 caches have 1440 flow and 88 SAM files, 720 B0 steps replay checked. Final300 teacher cache is pending.
Current initial clouds are train-only sparse calibrated SIFT, not author COLMAP dense; mark strict development protocol and do not claim paper reproduction.
Implement independent validation/evaluation and teacher-cache training adapter; empty test loader currently prevents direct unchanged train.py use. Implement full checkpoint/resume and compare it before running the development baseline.
Retain B0 FP32 original labels. Shared FP16 flow resize correction and FB-valid mask are separate controls. Never apply shared mask to original B0.
Meet Room original scenes discussion/trimming/vrheadset; cam_0 verified central test; 13 cameras, 1280x720 public calibration. Numeric camera sorting is required. Only lists/poses downloaded; videos and visual pose verification pending.
Do not run phase01 finalizer/update_handoff or phase02 prepare init to overwrite current state.
''',encoding='utf-8')
    print(json.dumps(dict(status='PASS',pngs=count,extracted_GiB=total_bytes/2**30,teacher_flow_files=1440,sam_files=88,
          report=str(ROOT/'reports/phase02_data_audit.md')),indent=2))
if __name__=='__main__':run()
