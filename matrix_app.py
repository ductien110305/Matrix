import streamlit as st
import numpy as np

# ===================== Helper Functions =====================
def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0

def dh_transform(d, theta_deg, a, alpha_deg):
    """DH transform theo thứ tự d, θ, a, α"""
    a = float(a)
    d = float(d)
    alpha = np.deg2rad(float(alpha_deg))
    theta = np.deg2rad(float(theta_deg))
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    T = np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ], dtype=float)
    return T

# ===================== App Layout =====================
st.set_page_config(page_title="Matrix & DH Calculator Pro", layout="centered")
st.title("📘 Matrix Calculator Pro (Streamlit Edition)")

tab1, tab2 = st.tabs(["📊 Matrix Calculator", "🤖 DH Table (d, θ, a, α)"])

# ===================== MATRIX TAB =====================
with tab1:
    st.header("Matrix Operations")
    st.markdown("Tạo ma trận A và B giống giao diện phần mềm chuyên nghiệp 🧮")

    cols = st.columns(3)
    rows = cols[0].number_input("Rows", min_value=1, max_value=6, value=3, key="rows")
    cols_n = cols[1].number_input("Cols", min_value=1, max_value=6, value=3, key="cols")
    create = cols[2].button("Create Matrices")

    if create:
        st.session_state.rows = int(rows)
        st.session_state.cols = int(cols_n)

    if "rows" in st.session_state and "cols" in st.session_state:
        rows, cols_n = st.session_state.rows, st.session_state.cols

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Matrix A")
            A = np.zeros((rows, cols_n))
            for i in range(rows):
                cols_row = st.columns(cols_n)
                for j in range(cols_n):
                    A[i, j] = cols_row[j].number_input(f"A{i}{j}", value=0.0, step=1.0, key=f"A_{i}_{j}")

        with colB:
            st.subheader("Matrix B")
            B = np.zeros((rows, cols_n))
            for i in range(rows):
                cols_row = st.columns(cols_n)
                for j in range(cols_n):
                    B[i, j] = cols_row[j].number_input(f"B{i}{j}", value=0.0, step=1.0, key=f"B_{i}_{j}")

        st.divider()
        col_ops = st.columns(5)
        ops = [
            ("A + B", A + B),
            ("A - B", A - B),
            ("A × B", np.dot(A, B) if A.shape[1] == B.shape[0] else "❌ Không hợp lệ"),
            ("det(A)", np.linalg.det(A) if A.shape[0] == A.shape[1] else "❌ A không vuông"),
            ("det(B)", np.linalg.det(B) if B.shape[0] == B.shape[1] else "❌ B không vuông"),
        ]

        inv_ops = st.columns(5)

        # --- Tính toán nghịch đảo và chuyển vị ---
        try:
            invA = np.linalg.inv(A) if A.shape[0] == A.shape[1] else "❌ A không vuông"
        except np.linalg.LinAlgError:
            invA = "❌ A suy biến (det = 0, không có nghịch đảo)"

        try:
            invB = np.linalg.inv(B) if B.shape[0] == B.shape[1] else "❌ B không vuông"
        except np.linalg.LinAlgError:
            invB = "❌ B suy biến (det = 0, không có nghịch đảo)"

        # --- Giải hệ phương trình ---
        try:
            solveAB = np.linalg.solve(A, B[:, [0]]) if A.shape[0] == A.shape[1] else "❌ Không giải được"
        except np.linalg.LinAlgError:
            solveAB = "❌ A suy biến (det = 0, không giải được)"

        # --- Gom kết quả vào danh sách ---
        extra_ops = [
            ("inv(A)", invA),
            ("inv(B)", invB),
            ("Aᵀ", A.T),
            ("Bᵀ", B.T),
            ("Solve A·x=B", solveAB),
        ]

        st.markdown("### 🔹 Kết quả:")
        result = None
        for name, res in ops:
            if col_ops[ops.index((name, res))].button(name):
                result = res
        for name, res in extra_ops:
            if inv_ops[extra_ops.index((name, res))].button(name):
                result = res

        if result is not None:
            st.write(result)

# ===================== DH TAB =====================
with tab2:
    st.header("Denavit–Hartenberg (DH) Parameters")
    st.markdown("Tính toán theo thứ tự: **d, θ, a, α**")

    n = st.number_input("Số khâu (links)", min_value=1, max_value=10, value=3)
    dh_list = []

    st.write("### Nhập thông số DH cho từng khâu:")
    for i in range(n):
        col1, col2, col3, col4 = st.columns(4)
        d = col1.number_input(f"d{i+1}", value=0.0, step=1.0, key=f"d{i}")
        theta = col2.number_input(f"θ{i+1} (deg)", value=0.0, step=1.0, key=f"theta{i}")
        a = col3.number_input(f"a{i+1}", value=0.0, step=1.0, key=f"a{i}")
        alpha = col4.number_input(f"α{i+1} (deg)", value=0.0, step=1.0, key=f"alpha{i}")
        dh_list.append((d, theta, a, alpha))

    if st.button("🧩 Compute DH Matrices"):
        st.latex(r"""
        A_i = R_z(\theta_i)T_z(d_i)T_x(a_i)R_x(\alpha_i) =
        \begin{bmatrix}
        \cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\
        \sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\
        0 & \sin\alpha_i & \cos\alpha_i & d_i \\
        0 & 0 & 0 & 1
        \end{bmatrix}
        """)
        T_total = np.eye(4)
        per_link = []
        try:
            for (d, theta, a, alpha) in dh_list:
                T = dh_transform(d, theta, a, alpha)
                per_link.append(T)
                T_total = T_total @ T

            for i, T in enumerate(per_link):
                st.markdown(f"### 🔸 Khâu {i+1}")
                st.write(T)

            st.success("✅ Ma trận tổng (Base → End-effector):")
            st.write(T_total)
        except Exception as e:
            st.error(f"Lỗi: {e}")

