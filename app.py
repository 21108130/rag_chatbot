import sys
import os
from pathlib import Path

Path("data/uploads").mkdir(parents=True, exist_ok=True)
Path("data/chroma_db").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from src.ui.app import main
main()
