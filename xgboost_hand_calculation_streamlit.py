import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(np.var(y_true), 0):
        return float("nan")
    return r2_score(y_true, y_pred)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": safe_r2(y_true, y_pred),
    }


def apply_preset(preset: str) -> dict:
    if preset == "Nhanh (khám phá)":
        return {"lambda": 0.8, "eta": 0.2, "iters": 10, "test": 20, "gamma": 0.0, "mcw": 1.0, "msl": 1}
    if preset == "Cân bằng (khuyến nghị)":
        return {"lambda": 1.0, "eta": 0.1, "iters": 20, "test": 25, "gamma": 0.2, "mcw": 2.0, "msl": 2}
    return {"lambda": 1.8, "eta": 0.05, "iters": 35, "test": 30, "gamma": 1.0, "mcw": 4.0, "msl": 3}


def candidate_thresholds(values: np.ndarray, max_bins: int = 64) -> np.ndarray:
    uniq = np.unique(values)
    if len(uniq) <= 1:
        return np.array([])
    if len(uniq) <= max_bins:
        return (uniq[:-1] + uniq[1:]) / 2.0

    quantiles = np.unique(np.quantile(values, np.linspace(0.05, 0.95, max_bins)))
    if len(quantiles) <= 1:
        return np.array([])
    return (quantiles[:-1] + quantiles[1:]) / 2.0


class TreeNode:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.weight = None
        self.gain = 0.0
        self.depth = 0


def find_best_split_by_grad_hess(
    x_train: pd.DataFrame,
    grad: np.ndarray,
    hess: np.ndarray,
    feature_candidates: list,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    min_samples_leaf: int,
) -> dict | None:
    total_g = float(np.sum(grad))
    total_h = float(np.sum(hess))
    best: dict | None = None

    for feat in feature_candidates:
        values = x_train[feat].to_numpy(dtype=float)
        thresholds = candidate_thresholds(values)
        if len(thresholds) == 0:
            continue

        for th in thresholds:
            left_mask = values < th
            right_mask = ~left_mask

            if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
                continue

            hl = float(np.sum(hess[left_mask]))
            hr = float(np.sum(hess[right_mask]))
            if hl < min_child_weight or hr < min_child_weight:
                continue

            gl = float(np.sum(grad[left_mask]))
            gr = float(np.sum(grad[right_mask]))

            gain = 0.5 * (
                (gl * gl) / (hl + reg_lambda)
                + (gr * gr) / (hr + reg_lambda)
                - (total_g * total_g) / (total_h + reg_lambda)
            ) - gamma

            if best is None or gain > best["gain"]:
                best = {
                    "feature": feat,
                    "threshold": float(th),
                    "gain": float(gain),
                    "left_mask": left_mask,
                    "right_mask": right_mask,
                }

    return best


def calc_leaf_weight(grad: np.ndarray, hess: np.ndarray, reg_lambda: float) -> float:
    if len(grad) == 0:
        return 0.0
    return float(-np.sum(grad) / (np.sum(hess) + reg_lambda))


def build_tree(
    x: pd.DataFrame,
    grad: np.ndarray,
    hess: np.ndarray,
    depth: int,
    max_depth: int,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
    min_samples_leaf: int,
    feature_candidates: list,
) -> TreeNode:
    node = TreeNode()
    node.depth = depth

    if depth >= max_depth or len(grad) < min_samples_leaf * 2:
        node.weight = calc_leaf_weight(grad, hess, reg_lambda)
        return node

    best_split = find_best_split_by_grad_hess(
        x_train=x,
        grad=grad,
        hess=hess,
        feature_candidates=feature_candidates,
        reg_lambda=reg_lambda,
        gamma=gamma,
        min_child_weight=min_child_weight,
        min_samples_leaf=min_samples_leaf,
    )

    if best_split is None or best_split["gain"] <= 0:
        node.weight = calc_leaf_weight(grad, hess, reg_lambda)
        return node

    node.feature = best_split["feature"]
    node.threshold = best_split["threshold"]
    node.gain = best_split["gain"]

    left_mask = best_split["left_mask"]
    right_mask = best_split["right_mask"]

    node.left = build_tree(
        x[left_mask].reset_index(drop=True),
        grad[left_mask],
        hess[left_mask],
        depth + 1,
        max_depth,
        reg_lambda,
        gamma,
        min_child_weight,
        min_samples_leaf,
        feature_candidates,
    )
    node.right = build_tree(
        x[right_mask].reset_index(drop=True),
        grad[right_mask],
        hess[right_mask],
        depth + 1,
        max_depth,
        reg_lambda,
        gamma,
        min_child_weight,
        min_samples_leaf,
        feature_candidates,
    )
    return node


def predict_tree(node: TreeNode, x: pd.DataFrame) -> np.ndarray:
    preds = np.zeros(len(x), dtype=float)
    for i, row in x.iterrows():
        current = node
        while current.weight is None:
            if row[current.feature] < current.threshold:
                current = current.left
            else:
                current = current.right
        preds[i] = current.weight
    return preds


def collect_split_logs(node: TreeNode, iteration: int, logs: list[dict]) -> None:
    if node is None:
        return
    if node.weight is not None:
        return

    logs.append(
        {
            "Iteration": iteration,
            "Depth": node.depth,
            "Feature": node.feature,
            "Threshold": node.threshold,
            "Gain": node.gain,
            "LeafWeight": np.nan,
        }
    )
    collect_split_logs(node.left, iteration, logs)
    collect_split_logs(node.right, iteration, logs)


def tree_to_records(node: TreeNode, iteration: int) -> pd.DataFrame:
    rows: list[dict] = []

    def walk(cur: TreeNode, node_id: str, parent_id: str) -> None:
        if cur.weight is not None:
            label = f"Lá dự đoán\\nĐiều chỉnh = {cur.weight:+.3f}"
            color_value = 0.0
        else:
            label = f"Chia theo {cur.feature}\\nNgưỡng < {cur.threshold:.3f}"
            color_value = float(cur.gain)

        rows.append(
            {
                "id": node_id,
                "parent": parent_id,
                "label": label,
                "depth": cur.depth,
                "color": color_value,
            }
        )

        if cur.weight is None:
            walk(cur.left, f"{node_id}L", node_id)
            walk(cur.right, f"{node_id}R", node_id)

    walk(node, f"iter_{iteration}_root", "")
    return pd.DataFrame(rows)


def tree_summary_lines(node: TreeNode, indent: str = "", branch: str = "Gốc") -> list[str]:
    if node is None:
        return []

    if node.weight is not None:
        return [f"{indent}{branch}: lá dự đoán, điều chỉnh = {node.weight:+.3f}"]

    lines = [f"{indent}{branch}: chia theo {node.feature} < {node.threshold:.3f}"]
    lines.extend(tree_summary_lines(node.left, indent + "  ", "Trái"))
    lines.extend(tree_summary_lines(node.right, indent + "  ", "Phải"))
    return lines


def tree_conclusion_text(node: TreeNode) -> str:
    if node is None:
        return "Cây này không có nội dung để diễn giải."

    if node.weight is not None:
        direction = "tăng" if node.weight > 0 else "giảm"
        return f"Cây này không tách thêm; nó chỉ điều chỉnh dự đoán theo hướng {direction} {abs(node.weight):.3f}."

    leaf_weights: list[float] = []

    def collect_leaves(cur: TreeNode) -> None:
        if cur is None:
            return
        if cur.weight is not None:
            leaf_weights.append(float(cur.weight))
            return
        collect_leaves(cur.left)
        collect_leaves(cur.right)

    collect_leaves(node)
    if leaf_weights:
        positive = sum(weight > 0 for weight in leaf_weights)
        negative = sum(weight < 0 for weight in leaf_weights)
        dominant_direction = "tăng" if positive >= negative else "giảm"
    else:
        dominant_direction = "tăng"

    root_desc = f"Cây ưu tiên tách theo {node.feature} tại ngưỡng {node.threshold:.3f}."
    if node.gain > 0:
        gain_desc = f" Điểm chia này giúp giảm lỗi với mức gain {node.gain:.4f}."
    else:
        gain_desc = ""

    return (
        f"{root_desc}{gain_desc}"
        f" Sau khi tách, các lá chủ yếu điều chỉnh dự đoán theo hướng {dominant_direction}."
    )


def compute_feature_importance(split_log_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    if split_log_df.empty:
        return pd.DataFrame(
            {
                "Feature": feature_cols,
                "Total Gain": np.zeros(len(feature_cols), dtype=float),
                "Split Count": np.zeros(len(feature_cols), dtype=int),
                "Avg Gain": np.zeros(len(feature_cols), dtype=float),
            }
        )

    gain_imp = (
        split_log_df[split_log_df["Feature"] != "(Không tách thêm)"]
        .groupby("Feature")["Gain"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Total Gain", "count": "Split Count"})
        .reset_index()
    )
    if not gain_imp.empty:
        gain_imp["Avg Gain"] = gain_imp["Total Gain"] / gain_imp["Split Count"]
    else:
        gain_imp = pd.DataFrame(columns=["Feature", "Total Gain", "Split Count", "Avg Gain"])

    used = set(gain_imp["Feature"]) if not gain_imp.empty else set()
    missing = [
        {"Feature": f, "Total Gain": 0.0, "Split Count": 0, "Avg Gain": 0.0}
        for f in feature_cols
        if f not in used
    ]
    if missing:
        gain_imp = pd.concat([gain_imp, pd.DataFrame(missing)], ignore_index=True)

    return gain_imp.sort_values("Total Gain", ascending=False).reset_index(drop=True)


def run_boosting_dynamic(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    n_iter: int,
    lr: float,
    reg_lambda: float,
    gamma: float,
    feature_candidates: list,
    max_depth: int,
    min_child_weight: float,
    min_samples_leaf: int,
    subsample: float,
    colsample: float,
    early_stopping_rounds: int,
    x_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame, list[TreeNode], dict]:
    init_pred = float(np.mean(y_train))
    train_pred = np.full(len(y_train), init_pred, dtype=float)
    test_pred = np.full(len(x_test), init_pred, dtype=float)
    val_pred = np.full(len(x_val), init_pred, dtype=float) if x_val is not None else None

    history_train = [train_pred.copy()]
    history_test = [test_pred.copy()]
    history_val = [val_pred.copy()] if val_pred is not None else None
    split_logs = []
    trees: list[TreeNode] = []
    best_val_mae = float("inf")
    best_iter = 0
    no_improve = 0
    early_stop_iter = 0
    rng = np.random.default_rng(42)

    for it in range(1, n_iter + 1):
        grad = -(y_train - train_pred)
        hess = np.ones(len(y_train), dtype=float)

        if subsample < 1.0:
            n_sample = max(int(len(x_train) * subsample), 2)
            sample_idx = rng.choice(len(x_train), n_sample, replace=False)
        else:
            sample_idx = np.arange(len(x_train))

        x_sub = x_train.iloc[sample_idx].reset_index(drop=True)
        grad_sub = grad[sample_idx]
        hess_sub = hess[sample_idx]

        if colsample < 1.0:
            n_cols = max(int(len(feature_candidates) * colsample), 1)
            col_sub = list(rng.choice(feature_candidates, n_cols, replace=False))
        else:
            col_sub = feature_candidates

        tree = build_tree(
            x=x_sub,
            grad=grad_sub,
            hess=hess_sub,
            depth=0,
            max_depth=max_depth,
            reg_lambda=reg_lambda,
            gamma=gamma,
            min_child_weight=min_child_weight,
            min_samples_leaf=min_samples_leaf,
            feature_candidates=col_sub,
        )

        train_update = predict_tree(tree, x_train.reset_index(drop=True))
        test_update = predict_tree(tree, x_test.reset_index(drop=True))
        val_update = predict_tree(tree, x_val.reset_index(drop=True)) if x_val is not None else None

        train_pred += lr * train_update
        test_pred += lr * test_update
        if val_pred is not None:
            val_pred += lr * val_update
        trees.append(tree)

        history_train.append(train_pred.copy())
        history_test.append(test_pred.copy())
        if history_val is not None:
            history_val.append(val_pred.copy())

        n_before = len(split_logs)
        collect_split_logs(tree, iteration=it, logs=split_logs)
        if len(split_logs) == n_before:
            split_logs.append(
                {
                    "Iteration": it,
                    "Depth": 0,
                    "Feature": "(Không tách thêm)",
                    "Threshold": np.nan,
                    "Gain": 0.0,
                    "LeafWeight": float(tree.weight if tree.weight is not None else 0.0),
                }
            )

        if val_pred is not None:
            current_mae = mean_absolute_error(y_val, val_pred)
            if current_mae < best_val_mae - 1e-6:
                best_val_mae = float(current_mae)
                best_iter = it
                no_improve = 0
            else:
                no_improve += 1

        if val_pred is not None and early_stopping_rounds > 0 and no_improve >= early_stopping_rounds:
            early_stop_iter = it
            break

    if early_stop_iter == 0:
        early_stop_iter = len(history_test) - 1

    if val_pred is None:
        best_iter = early_stop_iter
    meta = {
        "best_val_mae": best_val_mae if val_pred is not None else None,
        "best_iter": best_iter,
        "early_stop_iter": early_stop_iter,
        "stopped_early": early_stop_iter < n_iter,
    }

    val_history = np.array(history_val) if history_val is not None else None
    return init_pred, np.array(history_train), np.array(history_test), val_history, pd.DataFrame(split_logs), trees, meta


st.set_page_config(page_title="XGBoost Hand Calculation", layout="wide")
st.markdown(
    """
<style>
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
    border: 1px solid #e6ebf2;
    border-radius: 14px;
    padding: 12px 14px;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #e8edf4;
    border-radius: 12px;
}
.hint-card {
    border: 1px solid #e7ecf4;
    border-left: 4px solid #3b82f6;
    border-radius: 12px;
    padding: 10px 12px;
    background: #f9fbff;
}
</style>
""",
    unsafe_allow_html=True,
)
st.title("🚀 XGBoost Hand Calculation - Enhanced")
st.caption("Nhập/sửa dữ liệu trực tiếp, thêm dòng dữ liệu, so sánh baseline và theo dõi kết quả interactive.")

tab_data, tab_model, tab_analysis = st.tabs(["📊 Dữ liệu", "🎯 Mô hình", "🔬 Phân tích lỗi"])

default_df = pd.DataFrame(
    {
        "SQFT": [1000, 1200, 800, 1500, 1100, 1300, 900, 1400, 1050, 1250],
        "Rooms": [2, 3, 2, 4, 3, 3, 2, 4, 3, 3],
        "Age": [5, 10, 20, 2, 15, 8, 12, 3, 18, 7],
        "Price": [200000, 250000, 150000, 300000, 220000, 280000, 170000, 320000, 210000, 260000],
    }
)

with tab_data:
    st.subheader("Nhập dữ liệu")
    st.markdown('<div class="hint-card">Dữ liệu mẫu chỉ dùng để demo quy trình. Khi bạn thay dữ liệu, toàn bộ metric sẽ thay đổi theo dữ liệu mới.</div>', unsafe_allow_html=True)
    source = st.radio(
        "Nguồn dữ liệu",
        ["Dữ liệu mẫu (10 dòng)", "Upload CSV", "Tạo dữ liệu tổng hợp"],
        horizontal=True,
    )
    uploaded = st.file_uploader("Chọn file CSV", type=["csv"], disabled=source != "Upload CSV")

    if source == "Upload CSV" and uploaded is not None:
        base_df = pd.read_csv(uploaded)
        st.success(f"Đã tải {len(base_df)} dòng.")
    elif source == "Tạo dữ liệu tổng hợp":
        col_a, col_b, col_c = st.columns(3)
        n_synth = col_a.slider("Số dòng", 100, 2000, 500)
        n_feat = col_b.slider("Số features", 2, 8, 3)
        noise = col_c.slider("Noise level", 0, 100, 20)

        from sklearn.datasets import make_regression

        x_syn, y_syn = make_regression(
            n_samples=n_synth,
            n_features=n_feat,
            noise=noise,
            random_state=42,
        )
        cols = [f"Feature_{i + 1}" for i in range(n_feat)]
        base_df = pd.DataFrame(x_syn, columns=cols)
        base_df["Target"] = y_syn
        st.success(f"Đã tạo {n_synth} dòng với {n_feat} features.")
    else:
        base_df = default_df.copy()
        st.info("Bạn có thể chỉnh sửa trực tiếp dữ liệu mẫu và thêm dòng mới.")

    edited_df = st.data_editor(
        base_df,
        width="stretch",
        num_rows="dynamic",
        hide_index=True,
        key="editable_input_data",
    )

    st.write(f"Số dòng hiện tại: {len(edited_df)}")
    st.download_button(
        label="Tải dữ liệu đang sửa (.csv)",
        data=edited_df.to_csv(index=False).encode("utf-8"),
        file_name="edited_input_data.csv",
        mime="text/csv",
    )


analysis_payload = None

with tab_model:
    st.subheader("Cấu hình và kết quả")

    df = edited_df.copy()
    if df.empty:
        st.error("Dữ liệu rỗng. Vui lòng nhập ít nhất 4 dòng dữ liệu.")
        st.stop()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Cần ít nhất 2 cột số: 1 cột feature và 1 cột target.")
        st.stop()

    default_target_idx = numeric_cols.index("Price") if "Price" in numeric_cols else len(numeric_cols) - 1
    target_col = st.selectbox("Cột target", options=numeric_cols, index=default_target_idx)

    feature_cols = [c for c in numeric_cols if c != target_col]
    if not feature_cols:
        st.error("Không có cột feature hợp lệ sau khi chọn target.")
        st.stop()

    default_candidates = feature_cols.copy()
    split_candidates = st.multiselect(
        "Các cột được phép tìm split theo từng iteration",
        options=feature_cols,
        default=default_candidates,
    )
    if not split_candidates:
        st.error("Cần chọn ít nhất 1 cột để tìm split.")
        st.stop()

    preset_name = st.selectbox("Preset cấu hình", ["Cân bằng (khuyến nghị)", "Nhanh (khám phá)", "Thận trọng (ổn định)"])
    preset = apply_preset(preset_name)
    param_mode = st.radio("Mức điều chỉnh tham số", ["Cơ bản", "Nâng cao"], horizontal=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        lambda_val = st.slider("L2 (lambda)", 0.0, 5.0, float(preset["lambda"]), 0.1)
    with col2:
        eta = st.slider("Tốc độ học (eta)", 0.01, 1.0, float(preset["eta"]), 0.01)
    with col3:
        iterations = st.slider("Số iteration", 1, 60, int(preset["iters"]))
    with col4:
        test_ratio = st.slider("Test size (%)", 10, 50, int(preset["test"]), 5)

    max_depth = st.slider("Max depth mỗi tree", 1, 6, 2)

    gamma = float(preset["gamma"])
    min_child_weight = float(preset["mcw"])
    min_samples_leaf = int(preset["msl"])
    subsample = 1.0
    colsample = 1.0
    use_early_stop = True
    patience = 5
    if param_mode == "Nâng cao":
        col5, col6 = st.columns(2)
        with col5:
            gamma = st.slider("Gamma (phạt split nhỏ)", 0.0, 10.0, float(preset["gamma"]), 0.1)
        with col6:
            min_child_weight = st.slider("Min child weight", 1.0, 20.0, float(preset["mcw"]), 1.0)
        min_samples_leaf = st.slider("Min samples mỗi lá", 1, 20, int(preset["msl"]))
        col7, col8 = st.columns(2)
        with col7:
            subsample = st.slider("Row subsample", 0.4, 1.0, 0.8, 0.05)
        with col8:
            colsample = st.slider("Col subsample (colsample_bytree)", 0.4, 1.0, 0.8, 0.05)

    use_early_stop = st.checkbox("Early stopping", value=True)
    patience = st.slider("Patience (rounds)", 3, 15, 5) if use_early_stop else 0

    with st.expander("Giải thích logic thuật toán", expanded=False):
        st.markdown(
            """
1. Mỗi iteration mô hình tính gradient/hessian từ sai số hiện tại.
2. Tự tìm split tốt nhất trong các feature được chọn.
3. Tính leaf weight theo công thức có regularization lambda.
4. Cập nhật dự đoán với learning rate eta.

Lưu ý: Đây là phiên bản đơn giản hóa theo tinh thần XGBoost, chưa phải full production implementation.
"""
        )

    clean_df = df[feature_cols + [target_col]].dropna()
    if len(clean_df) < 4:
        st.error("Dữ liệu hợp lệ quá ít sau khi loại bỏ giá trị thiếu. Cần >= 4 dòng.")
        st.stop()

    x = clean_df[feature_cols]
    y = clean_df[target_col].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_ratio / 100, random_state=42
    )
    if patience and len(x_train) >= 10:
        x_fit, x_val, y_fit, y_val = train_test_split(
            x_train, y_train, test_size=0.25, random_state=43
        )
    else:
        x_fit, y_fit = x_train, y_train
        x_val, y_val = None, None
        if patience:
            st.info("Dữ liệu quá ít để tách validation; early stopping đã tắt. Test vẫn được giữ riêng.")
            patience = 0
    st.write(f"Train: {len(x_fit)} mẫu | Validation: {0 if x_val is None else len(x_val)} mẫu | Test: {len(x_test)} mẫu")
    actual_test_ratio = len(x_test) / len(clean_df)
    if actual_test_ratio < 0.15:
        st.warning("Tập test hiện khá nhỏ so với toàn bộ dữ liệu; metric có thể dao động mạnh.")
    elif actual_test_ratio < 0.25:
        st.info("Tập test ở mức trung bình; nên kiểm tra thêm nhiều lần chia dữ liệu để kết luận chắc chắn.")
    else:
        st.success("Tỷ lệ test tương đối ổn để tham khảo metric trên giao diện.")

    init_pred, history_train, history_test, history_val, split_log_df, trees, training_meta = run_boosting_dynamic(
        x_fit,
        y_fit,
        x_test,
        n_iter=iterations,
        lr=eta,
        reg_lambda=lambda_val,
        gamma=gamma,
        feature_candidates=split_candidates,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        colsample=colsample,
        early_stopping_rounds=patience,
        x_val=x_val,
        y_val=y_val,
    )

    selected_iteration = training_meta["best_iter"] if patience else len(trees)
    pred_train = history_train[selected_iteration]
    pred_test = history_test[selected_iteration]

    baseline_model = LinearRegression()
    baseline_model.fit(x_fit[split_candidates], y_fit)
    baseline_test_pred = baseline_model.predict(x_test[split_candidates])

    stump_model = DecisionTreeRegressor(max_depth=1, random_state=42)
    stump_model.fit(x_fit[split_candidates], y_fit)
    stump_test_pred = stump_model.predict(x_test[split_candidates])

    boost_train_metrics = metric_dict(y_fit, pred_train)
    boost_test_metrics = metric_dict(y_test, pred_test)
    lr_test_metrics = metric_dict(y_test, baseline_test_pred)
    stump_test_metrics = metric_dict(y_test, stump_test_pred)

    best_baseline_mae = min(lr_test_metrics["MAE"], stump_test_metrics["MAE"])
    effective_iterations = len(history_test) - 1

    st.markdown("### Chỉ số đánh giá")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE Train (Boosting)", f"{boost_train_metrics['MAE']:,.0f}")
    m2.metric("MAE Test (Boosting)", f"{boost_test_metrics['MAE']:,.0f}", delta=f"{best_baseline_mae - boost_test_metrics['MAE']:,.0f} so với baseline tốt nhất")
    m3.metric("RMSE Test (Boosting)", f"{boost_test_metrics['RMSE']:,.0f}")
    m4.metric("R2 Test (Boosting)", "N/A" if np.isnan(boost_test_metrics["R2"]) else f"{boost_test_metrics['R2']:.3f}")
    if training_meta["stopped_early"]:
        st.info(
            f"Early stopping tại iteration {training_meta['early_stop_iter']} | "
            f"Best MAE validation={training_meta['best_val_mae']:,.0f} ở iteration {training_meta['best_iter']}"
        )

    used_features_text = ", ".join(split_candidates)
    st.write(
        "Boosting hiện đã xây cây đệ quy theo gradient/hessian ở mỗi iteration, "
        f"max_depth={max_depth}, trong tập feature cho phép: {used_features_text}."
    )

    comparison_df = pd.DataFrame(
        {
            "Mô hình": ["Boosting động (gần XGBoost)", "Linear Regression", "Decision Stump (depth=1)"],
            "MAE Test": [boost_test_metrics["MAE"], lr_test_metrics["MAE"], stump_test_metrics["MAE"]],
            "RMSE Test": [boost_test_metrics["RMSE"], lr_test_metrics["RMSE"], stump_test_metrics["RMSE"]],
            "R2 Test": [boost_test_metrics["R2"], lr_test_metrics["R2"], stump_test_metrics["R2"]],
        }
    ).sort_values("MAE Test", ascending=True).reset_index(drop=True)
    comparison_df.insert(0, "Xếp hạng (MAE)", np.arange(1, len(comparison_df) + 1))
    st.markdown("### Bảng so sánh mô hình")
    st.dataframe(
        comparison_df.style.format({"MAE Test": "{:,.0f}", "RMSE Test": "{:,.0f}", "R2 Test": "{:.3f}"}),
        width="stretch",
    )

    st.markdown("### Lịch sử split theo iteration")
    st.dataframe(
        split_log_df.style.format(
            {
                "Depth": "{:,.0f}",
                "Threshold": "{:,.3f}",
                "Gain": "{:,.6f}",
                "LeafWeight": "{:,.4f}",
            }
        ),
        width="stretch",
    )

    st.markdown("### Feature Importance (Gain-based)")
    importance_df = compute_feature_importance(split_log_df, feature_cols)
    fig_imp = px.bar(
        importance_df,
        x="Total Gain",
        y="Feature",
        orientation="h",
        color="Avg Gain",
        color_continuous_scale="Blues",
        title="Feature Importance (Gain-based)",
        labels={"Total Gain": "Tổng Gain", "Feature": "Feature", "Avg Gain": "Avg Gain/Split"},
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, width="stretch")

    results_df = x_test.copy()
    results_df[target_col] = y_test
    results_df["Pred_Boosting"] = pred_test
    results_df["Pred_Baseline_LR"] = baseline_test_pred
    results_df["Pred_Stump"] = stump_test_pred
    results_df["Error_Boosting"] = y_test - pred_test
    results_df["Error_Baseline"] = y_test - baseline_test_pred
    results_df["Error_Stump"] = y_test - stump_test_pred
    results_df = results_df.reset_index(drop=True)
    results_df.insert(0, "Sample", np.arange(1, len(results_df) + 1))

    st.markdown("### Bảng kết quả test")
    st.dataframe(results_df, width="stretch")
    st.download_button(
        label="Tải kết quả dự đoán CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_results.csv",
        mime="text/csv",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Dự đoán vs Thực tế (hover để xem giá trị)")
        scatter_df = pd.DataFrame(
            {
                "Actual": y_test,
                "Boosting": pred_test,
                "BaselineLR": baseline_test_pred,
                "Stump": stump_test_pred,
            }
        )
        fig_scatter = px.scatter(
            scatter_df,
            x="Actual",
            y="Boosting",
            hover_data={"BaselineLR": ":.2f", "Stump": ":.2f", "Actual": ":.2f", "Boosting": ":.2f"},
            title="So sánh dự đoán Boosting và giá trị thực",
        )
        min_val = float(min(scatter_df["Actual"].min(), scatter_df["Boosting"].min(), scatter_df["BaselineLR"].min()))
        max_val = float(max(scatter_df["Actual"].max(), scatter_df["Boosting"].max(), scatter_df["BaselineLR"].max()))
        fig_scatter.add_shape(
            type="line",
            x0=min_val,
            y0=min_val,
            x1=max_val,
            y1=max_val,
            line=dict(color="firebrick", dash="dash"),
        )
        st.plotly_chart(fig_scatter, width="stretch")

    with c2:
        st.markdown("### Lỗi hội tụ qua iteration (hover)")
        train_mae_curve = [mean_absolute_error(y_fit, p) for p in history_train]
        test_mae_curve = [mean_absolute_error(y_test, p) for p in history_test]
        val_mae_curve = [mean_absolute_error(y_val, p) for p in history_val] if history_val is not None else None

        fig_curve = go.Figure()
        fig_curve.add_trace(
            go.Scatter(
                x=list(range(len(train_mae_curve))),
                y=train_mae_curve,
                mode="lines+markers",
                name="Train MAE",
            )
        )
        fig_curve.add_trace(
            go.Scatter(
                x=list(range(len(test_mae_curve))),
                y=test_mae_curve,
                mode="lines+markers",
                name="Test MAE (chỉ báo cáo)",
            )
        )
        if val_mae_curve is not None:
            fig_curve.add_trace(go.Scatter(
                x=list(range(len(val_mae_curve))), y=val_mae_curve,
                mode="lines+markers", name="Validation MAE (chọn iteration)"
            ))
        fig_curve.add_hline(y=lr_test_metrics["MAE"], line_dash="dot", annotation_text="LR baseline MAE")
        fig_curve.add_hline(y=stump_test_metrics["MAE"], line_dash="dot", annotation_text="Stump baseline MAE")
        if training_meta["stopped_early"]:
            fig_curve.add_vrect(
                x0=training_meta["early_stop_iter"],
                x1=iterations,
                fillcolor="orange",
                opacity=0.08,
                annotation_text="Early stopped",
                annotation_position="top left",
                line_width=0,
            )
        if val_mae_curve is not None and training_meta["best_iter"] > 0:
            fig_curve.add_annotation(
                x=training_meta["best_iter"],
                y=training_meta["best_val_mae"],
                text=f"Best validation: {training_meta['best_val_mae']:,.0f}",
                arrowhead=2,
                font=dict(color="green"),
            )
        fig_curve.update_layout(xaxis_title="Iteration", yaxis_title="MAE")
        st.plotly_chart(fig_curve, width="stretch")

    st.markdown("### Cấu trúc tree theo iteration")
    if trees:
        tree_iter = st.slider("Chọn cây để xem", 1, len(trees), len(trees))
        selected_tree = trees[tree_iter - 1]
        summary_lines = tree_summary_lines(selected_tree)
        st.markdown("#### Tóm tắt quyết định của cây")
        st.code("\n".join(summary_lines), language="text")
        st.info(tree_conclusion_text(selected_tree))
        st.caption("Phần này nói cây đã chia dữ liệu như thế nào và mỗi lá đang cộng/trừ bao nhiêu vào dự đoán. Chi tiết kỹ thuật vẫn có trong biểu đồ bên dưới.")
        tree_df = tree_to_records(selected_tree, tree_iter)
        fig_tree = px.treemap(
            tree_df,
            names="label",
            ids="id",
            parents="parent",
            color="color",
            color_continuous_scale="Blues",
            title=f"Cấu trúc cây tại iteration {tree_iter}",
        )
        with st.expander("Xem biểu đồ kỹ thuật"):
            st.plotly_chart(fig_tree, width="stretch")

    st.markdown("### Chi tiết theo iteration")
    iter_step = st.slider("Chọn iteration để xem", 0, effective_iterations, effective_iterations)
    iter_df = pd.DataFrame(
        {
            "Sample": np.arange(1, len(y_test) + 1),
            "Actual": y_test,
            "Pred_Boosting": history_test[iter_step],
            "Error": y_test - history_test[iter_step],
        }
    )
    st.dataframe(iter_df, width="stretch")

    st.caption(
        "Công thức cốt lõi: l = 1/2 (y - y_hat)^2, g = -(y - y_hat), h = 1, "
        "w = -Sum(g) / (Sum(h) + lambda). Ở phần tree, con số hiển thị là quyết định chia nhánh và mức điều chỉnh dự đoán, "
        "không phải một diễn giải nghiệp vụ riêng của dữ liệu."
    )

    analysis_payload = {
        "y_test": y_test,
        "pred_test": pred_test,
    }

with tab_analysis:
    st.subheader("Phân tích residual")
    if analysis_payload is None:
        st.info("Chạy tab Mô hình trước để tạo dự đoán và residual.")
    else:
        y_test = analysis_payload["y_test"]
        pred_test = analysis_payload["pred_test"]
        residual = y_test - pred_test

        fig_res = px.histogram(
            x=residual,
            nbins=20,
            title="Phân phối residual (Test set)",
            labels={"x": "Residual (Actual - Predicted)"},
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, width="stretch")

        fig_rv = px.scatter(
            x=pred_test,
            y=residual,
            title="Residual vs Predicted",
            labels={"x": "Predicted", "y": "Residual"},
        )
        fig_rv.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rv, width="stretch")
