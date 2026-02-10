source .venv/bin/activate && python - <<'PY'
import sys, traceback
print('Python', sys.version.replace('\n',' '))
try:
    import torch
    print('Torch', torch.__version__, 'CUDA', torch.version.cuda, 'cuda.is_available()', torch.cuda.is_available())
except Exception as e:
    print('Torch import failed:', e)
try:
    import k2
    print('k2 imported:', getattr(k2, '__version__', 'unknown'), getattr(k2, '__file__', 'no __file__'))
except Exception as e:
    print('k2 import failed:', e)
try:
    import _k2
    print('_k2 imported OK')
except Exception as e:
    print('_k2 import failed:', e)
PY