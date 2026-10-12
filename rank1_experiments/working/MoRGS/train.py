"""Default entry for the six-attribute, RGB-only online baseline."""
import sys
import importlib.util
from pathlib import Path
_baseline = Path(__file__).resolve().parents[3] / 'morgs_clean_baseline'
sys.path.insert(0, str(_baseline))
_spec = importlib.util.spec_from_file_location('clean_online_training', _baseline / 'train.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
main = _module.main
if __name__ == '__main__':
    main()
