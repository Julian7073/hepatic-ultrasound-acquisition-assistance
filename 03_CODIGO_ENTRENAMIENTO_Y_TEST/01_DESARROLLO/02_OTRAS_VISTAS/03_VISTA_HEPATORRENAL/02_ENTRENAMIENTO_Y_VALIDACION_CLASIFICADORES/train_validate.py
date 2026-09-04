"""Development-only comparison and fitting for hepatorrenal; excludes P005."""
from pathlib import Path
import runpy
import sys
REPO = next(p for p in Path(__file__).resolve().parents if (p/'Codigos_DINO_Experimental').is_dir())
sys.argv = [str(REPO/'Codigos_DINO_Experimental/scripts/14_reproduce_development_only.py'), '--views', 'hepatorrenal', *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name='__main__')
