"""internal_test: oblicua; delegates to the common, development-only replay."""
from pathlib import Path
import runpy
import sys
REPO = next(p for p in Path(__file__).resolve().parents if (p/'Codigos_DINO_Experimental').is_dir())
sys.argv = [str(REPO/'Codigos_DINO_Experimental/scripts/15_reproduce_frozen_stage.py'), '--stage', 'internal_test', '--view', 'oblicua', *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name='__main__')
