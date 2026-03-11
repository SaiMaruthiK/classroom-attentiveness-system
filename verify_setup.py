"""
Quick environment verification script.
Run: python verify_setup.py
"""

import sys
print(f"Python: {sys.version}")
print("-" * 50)

checks = [
    ("numpy", "import numpy; print(f'  numpy {numpy.__version__}')"),
    ("opencv", "import cv2; print(f'  opencv {cv2.__version__}')"),
    ("torch", "import torch; print(f'  torch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"),
    ("ultralytics (YOLOv8)", "import ultralytics; print(f'  ultralytics {ultralytics.__version__}')"),
    ("mediapipe", "import mediapipe; print(f'  mediapipe {mediapipe.__version__}')"),
    ("fer (emotion)", "from fer import FER; print('  FER OK')" ),
    ("fer (emotion)", "from fer import FER; print('  FER OK')"),
    ("fastapi", "import fastapi; print(f'  fastapi {fastapi.__version__}')"),
    ("streamlit", "import streamlit; print(f'  streamlit {streamlit.__version__}')"),
    ("sqlalchemy", "import sqlalchemy; print(f'  sqlalchemy {sqlalchemy.__version__}')"),
    ("plotly", "import plotly; print(f'  plotly {plotly.__version__}')"),
    ("deep_sort_realtime", "from deep_sort_realtime.deepsort_tracker import DeepSort; print('  DeepSORT OK')"),
]

all_ok = True
for name, code in checks:
    try:
        exec(code)
        print(f"✅ {name}")
    except Exception as e:
        print(f"❌ {name}: {e}")
        all_ok = False

print("-" * 50)
if all_ok:
    print("✅ All dependencies verified! You're ready to run.")
else:
    print("⚠️  Some packages missing. Run: pip install -r requirements.txt")
