"""Smoke test of app.py without Streamlit installed: a stub `streamlit` module returns each widget's default value,
so every branch of the dashboard code runs against the real engine.  Run:  python -m tests.test_app_smoke"""
import sys
import types
from contextlib import contextmanager


class _Ctx:
    def __init__(self, log): self.log = log
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def number_input(self, label, min_value=None, max_value=None, value=0, step=1, **k): return value
    def selectbox(self, label, options, index=0, **k): return list(options)[index]
    def __getattr__(self, name): return lambda *a, **k: self.log.append(name)


def make_stub(log):
    st = types.ModuleType("streamlit")
    def cache_resource(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]
    st.cache_resource = cache_resource
    for n in ("set_page_config", "title", "caption", "subheader", "header", "markdown", "write", "dataframe", "pyplot",
              "warning", "info", "error", "success"):
        setattr(st, n, (lambda name: lambda *a, **k: log.append(name))(n))
    st.stop = lambda: (_ for _ in ()).throw(SystemExit("st.stop() called"))
    st.columns = lambda n: [_Ctx(log) for _ in range(n)]
    st.tabs = lambda labels: [_Ctx(log) for _ in labels]
    class Sidebar:
        header = lambda self, *a, **k: None
        radio = lambda self, label, options, index=0, **k: options[index]
        text_input = lambda self, label, value="", **k: value
        selectbox = lambda self, label, options, index=0, **k: list(options)[index]
        slider = lambda self, label, lo, hi, value, **k: value
        number_input = lambda self, label, min_value=None, max_value=None, value=0, step=1, **k: value
        select_slider = lambda self, label, options, value, **k: value
    st.sidebar = Sidebar()
    st.metric = lambda *a, **k: log.append("metric")
    return st


def main():
    log = []
    sys.modules["streamlit"] = make_stub(log)
    import app
    # widgets inside the tabs use st.number_input / st.selectbox directly
    st = sys.modules["streamlit"]
    st.number_input = lambda label, min_value=None, max_value=None, value=0, step=1, **k: value
    st.selectbox = lambda label, options, index=0, **k: list(options)[index]
    app.st = st
    app.main()
    assert log.count("pyplot") >= 6 and "dataframe" in log and "metric" in log, log
    print("app smoke test passed | calls:", len(log))


if __name__ == "__main__":
    main()
