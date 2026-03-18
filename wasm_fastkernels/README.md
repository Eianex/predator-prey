# wasm-fastkernels

A tiny CPython extension that replaces the Numba kernels for browser builds.

## Desktop build

```bash
python -m pip install -U build
python -m build --wheel
python -m pip install dist/wasm_fastkernels-0.1.0-*.whl
```

## WebAssembly build for pygbag's CPython runtime

Use pygame-web's python-wasm-sdk, not Pyodide.

```bash
git clone https://github.com/pygame-web/python-wasm-sdk /opt/python-wasm-sdk
/opt/python-wasm-sdk/python3-wasm -m build --wheel --no-isolation .
```

That produces a wasm wheel for pygbag's CPython runtime.

## App integration

In your main.py, choose the backend by platform:

```python
import sys

if sys.platform == "emscripten":
    from wasm_kernels import (
        solve_collisions_same_species,
        collect_nearby_indices,
        nearest_alive_index,
        nearest_alive_grown_index,
    )
else:
    from numba_collision_kernel import solve_collisions_same_species
    from numba_search_kernel import (
        collect_nearby_indices,
        nearest_alive_index,
        nearest_alive_grown_index,
    )
```

Then make sure the built wasm wheel is installed into the web runtime or vendored with your pygbag build.
