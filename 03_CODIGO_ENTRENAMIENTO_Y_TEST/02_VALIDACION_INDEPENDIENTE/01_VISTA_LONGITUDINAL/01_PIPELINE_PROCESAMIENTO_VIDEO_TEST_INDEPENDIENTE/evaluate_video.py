"""Delegate to the frozen longitudinal independent-video evaluator."""
from pathlib import Path
import runpy
import sys
REPO = next(p for p in Path(__file__).resolve().parents if (p/'Codigos_Pipeline_Experimental_Segmentacion').is_dir())
path = REPO/'Codigos_Pipeline_Experimental_Segmentacion/evaluate_p005_longitudinal_final.py'
sys.path.insert(0, str(path.parent))
sys.argv = [str(path), *sys.argv[1:]]
runpy.run_path(str(path), run_name='__main__')
