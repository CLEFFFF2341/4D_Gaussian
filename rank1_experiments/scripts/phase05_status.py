"""Read-only CPU status for the detached phase05 queue."""
import ctypes
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phase05.common import read

p=ROOT/'manifests/phase05_progress.json'
if not p.exists():print(json.dumps(dict(status='NOT_LAUNCHED')));sys.exit(0)
d=read(p);pid=d['pid']
kernel=ctypes.windll.kernel32
kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.OpenProcess.restype=ctypes.c_void_p
kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
kernel.CloseHandle.argtypes=[ctypes.c_void_p]
handle=kernel.OpenProcess(0x1000,False,pid);active=False
if handle:
    code=ctypes.c_ulong();kernel.GetExitCodeProcess(handle,ctypes.byref(code));active=code.value==259;kernel.CloseHandle(handle)
print(json.dumps(dict(status=d['status'],runner_pid=pid,runner_alive=active,
      updated=d['updated'],active=d['active'],active_complete_frame=d['active_complete_frame'],
      completed_search=f"{d['completed_search_jobs']}/24",completed_f0=sum(r['method']=='F0' for r in d['completed']),
      report=str(ROOT/'reports/phase05_go_no_go.md')),ensure_ascii=False))
