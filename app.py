# app.py
import streamlit as st
import joblib
import numpy as np
import tempfile, os
from pathlib import Path

# metrics libs
from radon.complexity import cc_visit
from radon.metrics import h_visit
import lizard

MODEL_FILENAME = "final_stacked_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        raise FileNotFoundError(f"{MODEL_FILENAME} not found. Put it in repo root.")
    return joblib.load(MODEL_FILENAME)

model = load_model()

def extract_basic(code_text):
    lines = code_text.splitlines()
    loc = len(lines)
    blank = sum(1 for l in lines if not l.strip())
    comments = sum(1 for l in lines if l.strip().startswith("#") or l.strip().startswith("//"))
    return {"loc": loc, "blank": blank, "comments": comments}

def extract_radon(code_text):
    try:
        cc = cc_visit(code_text)
        cyclomatic = float(np.mean([c.complexity for c in cc])) if cc else 0.0
    except Exception:
        cyclomatic = 0.0
    try:
        hal = list(h_visit(code_text))
        hal_totals = [h.total for h in hal] if hal else []
        hal_total = float(np.mean(hal_totals)) if hal_totals else 0.0
        uniq_ops = float(np.mean([h.distinct_operators for h in hal])) if hal else 0.0
        uniq_opnds = float(np.mean([h.distinct_operands for h in hal])) if hal else 0.0
    except Exception:
        hal_total = uniq_ops = uniq_opnds = 0.0
    return {"cyclomatic": cyclomatic, "hal_total": hal_total, "uniq_ops": uniq_ops, "uniq_opnds": uniq_opnds}

def extract_lizard(path):
    try:
        res = lizard.analyze_file(path)
        n_funcs = len(res.function_list)
        func_nloc = sum(f.nloc for f in res.function_list)
    except Exception:
        n_funcs = 0
        func_nloc = 0
    return {"n_funcs": n_funcs, "func_nloc": func_nloc}

def build_feature_vector(m):
    # EDIT this order to match the exact columns your model expects.
    # This is a best-effort mapping of common NASA-like features.
    fv = [
        m.get("loc",0),
        m.get("cyclomatic",0),
        m.get("hal_total",0),
        m.get("uniq_ops",0),
        m.get("n_funcs",0),
        m.get("hal_total",0),   # v
        m.get("func_nloc",0),   # l
        m.get("uniq_opnds",0),  # d
        m.get("uniq_ops",0),    # i
        m.get("hal_total",0),   # e
        0.0, 0.0, 0.0,
        m.get("comments",0),
        m.get("blank",0),
        0.0, m.get("uniq_ops",0), m.get("uniq_opnds",0),
        m.get("hal_total",0), m.get("uniq_opnds",0),
        m.get("func_nloc",0), 0.0, 0.0, 0.0, 0.0, 0.0,
        (m.get("comments",0)/(m.get("loc",1) or 1)),
        (m.get("blank",0)/(m.get("loc",1) or 1))
    ]
    return np.array(fv, dtype=float).reshape(1, -1)

st.set_page_config(page_title="Software Defect Predictor", layout="centered")
st.title("🧠 Software Defect Predictor — Upload code file")

st.write("Upload a Python (.py) or Java (.java) file. The app extracts metrics and predicts buggy / non-buggy.")

uploaded = st.file_uploader("Upload .py or .java file", type=["py","java"])
if uploaded:
    ext = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        code = open(tmp_path, "r", encoding="utf-8", errors="ignore").read()
        basic = extract_basic(code)
        rad = extract_radon(code)
        lz = extract_lizard(tmp_path)

        metrics = {}
        metrics.update(basic)
        metrics.update(rad)
        metrics.update(lz)
        metrics["comments"] = basic.get("comments",0)
        metrics["blank"] = basic.get("blank",0)

        st.subheader("Extracted metrics (sample)")
        st.write({
            "LOC": metrics["loc"],
            "Cyclomatic (avg)": round(metrics.get("cyclomatic",0),2),
            "Functions": metrics.get("n_funcs",0),
            "Comments": metrics.get("comments",0),
            "Blank lines": metrics.get("blank",0)
        })

        fv = build_feature_vector(metrics)
        pred = model.predict(fv)[0]
        proba = model.predict_proba(fv)[0][1] if hasattr(model,"predict_proba") else None

        if pred == 1:
            st.error("⚠️ Predicted: DEFECTIVE")
        else:
            st.success("✅ Predicted: NOT DEFECTIVE")

        if proba is not None:
            st.info(f"Model probability (defect): {proba:.2f}")

    except Exception as e:
        st.error("Error: " + str(e))
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

