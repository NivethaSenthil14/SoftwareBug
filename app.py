import streamlit as st
import joblib
import numpy as np
import tempfile
import os
from radon.complexity import cc_visit
from radon.metrics import h_visit
import lizard

# ----------------------------
# Load trained stacked model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("final_stacked_model.pkl")

model = load_model()

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="Software Defect Prediction", page_icon="🧠", layout="wide")

st.title("🧠 Software Defect Prediction System")
st.markdown("Upload a **Python/Java** source file to predict whether it contains software defects.")

uploaded_file = st.file_uploader("📂 Upload your code file", type=["py", "java"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
        tmp.write(uploaded_file.read())
        code_path = tmp.name

    try:
        with open(code_path, 'r') as f:
            code = f.read()

        # ----------------------------
        # 🧩 Feature Extraction
        # ----------------------------
        loc = len(code.splitlines())
        blank = sum(1 for line in code.splitlines() if not line.strip())
        comments = sum(1 for line in code.splitlines() if line.strip().startswith("#") or line.strip().startswith("//"))

        # Cyclomatic complexity (v(g))
        cc_results = cc_visit(code)
        v_g = np.mean([r.complexity for r in cc_results]) if cc_results else 1.0

        # Halstead metrics
        h_metrics = h_visit(code)
        if h_metrics:
            total_ops = np.mean([h.total for h in h_metrics])
            uniq_ops = np.mean([h.distinct_operators for h in h_metrics])
            uniq_opnd = np.mean([h.distinct_operands for h in h_metrics])
        else:
            total_ops = uniq_ops = uniq_opnd = 1.0

        # Lizard metrics (branch count, functions)
        lizard_result = lizard.analyze_file(code_path)
        branch_count = sum(f.nloc for f in lizard_result.function_list)
        n_funcs = len(lizard_result.function_list)

        # Derived ratios
        comment_density = comments / loc if loc else 0
        blank_ratio = blank / loc if loc else 0
        branch_density = branch_count / loc if loc else 0

        # ----------------------------
        # 🧠 Create feature vector
        # ----------------------------
        # NOTE: fill in approximate defaults for missing NASA-style features
        features = np.array([[loc, v_g, 1.0, 1.0, n_funcs, total_ops, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, comments, blank, 1.0, uniq_ops, uniq_opnd,
                              total_ops, uniq_opnd, branch_count, 1.0, branch_density,
                              1.0, 1.0, 1.0, comment_density, blank_ratio]])

        # ----------------------------
        # 🧾 Display metrics
        # ----------------------------
        st.subheader("📊 Extracted Software Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Lines of Code", loc)
        col2.metric("Comments", comments)
        col3.metric("Blank Lines", blank)
        col1.metric("Cyclomatic Complexity", round(v_g, 2))
        col2.metric("Branch Count", branch_count)
        col3.metric("Functions", n_funcs)

        # ----------------------------
        # 🔮 Prediction
        # ----------------------------
        prediction = model.predict(features)
        prob = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else None

        if prediction[0] == 1:
            st.error("⚠️ This code is predicted to be **DEFECTIVE**.")
        else:
            st.success("✅ This code is predicted to be **NON-DEFECTIVE**.")

        if prob is not None:
            st.info(f"Model Confidence: {prob*100:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Error during analysis: {e}")

    finally:
        os.remove(code_path)
