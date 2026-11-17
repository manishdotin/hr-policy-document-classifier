import py_compile


def test_streamlit_entrypoint_compiles():
    py_compile.compile("app/main.py", doraise=True)


def test_training_and_inference_modules_compile():
    py_compile.compile("app/train.py", doraise=True)
    py_compile.compile("app/inference.py", doraise=True)
