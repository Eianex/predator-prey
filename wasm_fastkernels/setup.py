from setuptools import Extension, setup

ext_modules = [
    Extension(
        "_fastkernels",
        sources=["_fastkernels.c"],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="wasm-fastkernels",
    version="0.1.0",
    py_modules=["wasm_kernels"],
    ext_modules=ext_modules,
)
