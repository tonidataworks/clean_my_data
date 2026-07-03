
import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="clean_my_data",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.20), transparent 28rem),
            radial-gradient(circle at top right, rgba(99, 102, 241, 0.18), transparent 30rem),
            linear-gradient(135deg, #f8fafc 0%, #eef2ff 48%, #ecfeff 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 55%, #134e4a 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 48%, #0f766e 100%);
        color: #ffffff;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.25);
        margin-bottom: 1.1rem;
    }

    .hero h1 {
        font-size: 2.35rem;
        margin: 0 0 0.35rem 0;
        letter-spacing: 0;
    }

    .hero p {
        font-size: 1.02rem;
        margin: 0;
        color: #dbeafe;
        max-width: 920px;
    }

    .metric-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(148, 163, 184, 0.30);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.82);
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }

    .metric-label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.65rem;
        color: #0f172a;
        font-weight: 750;
    }

    .section-note {
        padding: 0.85rem 1rem;
        border-left: 4px solid #14b8a6;
        border-radius: 10px;
        background: rgba(240, 253, 250, 0.90);
        color: #115e59;
        margin: 0.4rem 0 1rem 0;
    }

    div[data-testid="stTabs"] button {
        font-weight: 700;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 0;
        font-weight: 700;
    }
</style>
"""


st.markdown(APP_CSS, unsafe_allow_html=True)


def normalise_column_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "column"


def make_unique_columns(columns):
    seen = {}
    clean_columns = []

    for column in columns:
        base = normalise_column_name(column)
        count = seen.get(base, 0)
        clean_columns.append(base if count == 0 else f"{base}_{count + 1}")
        seen[base] = count + 1

    return clean_columns


def load_dataset(uploaded_file, selected_sheet=None):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, sheet_name=selected_sheet)

    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def get_column_groups(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    text_cols = [
        col
        for col in categorical_cols
        if df[col].astype(str).str.len().median() >= 20
    ]
    return numeric_cols, categorical_cols, date_cols, text_cols


def missing_summary(df):
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": (df.isna().mean().values * 100).round(2),
            "dtype": df.dtypes.astype(str).values,
            "unique_values": df.nunique(dropna=True).values,
        }
    )
    return summary.sort_values("missing_percent", ascending=False)


def outlier_summary(df, numeric_cols):
    rows = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            lower, upper = q1, q3
            count = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = ((series < lower) | (series > upper)).sum()

        rows.append(
            {
                "column": col,
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "outlier_count": int(count),
                "outlier_percent": round(count / len(df) * 100, 2),
            }
        )

    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned_data")
    return output.getvalue()


def apply_missing_strategy(df, columns, strategy, custom_value=None):
    updated = df.copy()

    for col in columns:
        if strategy == "Drop rows with missing values":
            updated = updated.dropna(subset=[col])
        elif strategy == "Fill with mean":
            updated[col] = updated[col].fillna(updated[col].mean())
        elif strategy == "Fill with median":
            updated[col] = updated[col].fillna(updated[col].median())
        elif strategy == "Fill with mode":
            mode_values = updated[col].mode(dropna=True)
            if not mode_values.empty:
                updated[col] = updated[col].fillna(mode_values.iloc[0])
        elif strategy == "Fill with zero":
            updated[col] = updated[col].fillna(0)
        elif strategy == "Fill with custom value":
            updated[col] = updated[col].fillna(custom_value)

    return updated


def cap_outliers_iqr(df, columns):
    updated = df.copy()

    for col in columns:
        series = updated[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        updated[col] = updated[col].clip(lower=lower, upper=upper)

    return updated


def remove_outliers_iqr(df, columns):
    updated = df.copy()

    for col in columns:
        series = updated[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        updated = updated[(updated[col].isna()) | ((updated[col] >= lower) & (updated[col] <= upper))]

    return updated


def render_metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def reset_working_data():
    st.session_state.working_df = st.session_state.raw_df.copy()
    st.session_state.cleaning_log = []


def log_action(action):
    st.session_state.cleaning_log.append(action)


st.markdown(
    """
    <div class="hero">
        <h1>clean_my_data</h1>
        <p>
            Upload a CSV or Excel file, diagnose data quality issues, clean the dataset,
            explore patterns, and export a cleaner version for analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.title("Control Panel")
    st.caption("Upload data, apply cleaning steps, and export your finished dataset.")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
    )

    selected_sheet = None
    if uploaded_file is not None and uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        excel_file = pd.ExcelFile(uploaded_file)
        selected_sheet = st.selectbox("Choose Excel sheet", excel_file.sheet_names)
        uploaded_file.seek(0)

    st.divider()
    st.subheader("Output")

    if "working_df" in st.session_state:
        st.download_button(
            "Download cleaned CSV",
            data=convert_df_to_csv(st.session_state.working_df),
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.download_button(
            "Download cleaned Excel",
            data=convert_df_to_excel(st.session_state.working_df),
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        if st.button("Reset to original data", use_container_width=True):
            reset_working_data()
            st.success("Working dataset reset.")
    else:
        st.info("Upload a dataset to unlock downloads.")


if uploaded_file is None:
    left, right = st.columns([1.1, 0.9])

    with left:
        st.subheader("What this app does")
        st.markdown(
            """
            This app is built for the first stage of any data project: understanding and
            cleaning the dataset. It supports common cleaning tasks such as missing-value
            handling, duplicate removal, data type conversion, text cleaning, outlier
            treatment, and categorical encoding.
            """
        )

        st.markdown(
            """
            <div class="section-note">
                Start by uploading a CSV or Excel file in the sidebar.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Suggested workflow")
        st.markdown(
            """
            1. Upload your dataset.
            2. Inspect the data profile.
            3. Clean missing values, duplicates, text, and outliers.
            4. Explore the cleaned data visually.
            5. Download the cleaned file.
            """
        )

    st.stop()


try:
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}-{selected_sheet}"

    if st.session_state.get("current_file_id") != current_file_id:
        uploaded_file.seek(0)
        raw_df = load_dataset(uploaded_file, selected_sheet)
        st.session_state.current_file_id = current_file_id
        st.session_state.raw_df = raw_df.copy()
        st.session_state.working_df = raw_df.copy()
        st.session_state.cleaning_log = []

except Exception as exc:
    st.error(f"Could not load the file: {exc}")
    st.stop()


df = st.session_state.working_df
numeric_cols, categorical_cols, date_cols, text_cols = get_column_groups(df)

rows, cols = df.shape
missing_cells = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
with metric_1:
    render_metric_card("Rows", f"{rows:,}")
with metric_2:
    render_metric_card("Columns", f"{cols:,}")
with metric_3:
    render_metric_card("Missing Cells", f"{missing_cells:,}")
with metric_4:
    render_metric_card("Duplicate Rows", f"{duplicate_rows:,}")


overview_tab, cleaning_tab, eda_tab, export_tab = st.tabs(
    ["Data Profile", "Cleaning Lab", "Visual EDA", "Export"]
)


with overview_tab:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Column Types")
        type_table = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": df.dtypes.astype(str),
                "non_null_count": df.notna().sum().values,
                "unique_values": df.nunique(dropna=True).values,
            }
        )
        st.dataframe(type_table, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Missing Value Summary")
        st.dataframe(missing_summary(df), use_container_width=True, hide_index=True)

    st.subheader("Descriptive Statistics")
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
    else:
        st.info("No numeric columns were detected for descriptive statistics.")

    st.subheader("Outlier Scan")
    if numeric_cols:
        st.dataframe(outlier_summary(df, numeric_cols), use_container_width=True, hide_index=True)
    else:
        st.info("No numeric columns were detected for outlier scanning.")


with cleaning_tab:
    st.subheader("Cleaning Tools")
    st.markdown(
        """
        <div class="section-note">
            Cleaning actions update the working dataset. Use the reset button in the sidebar
            if you want to return to the original upload.
        </div>
        """,
        unsafe_allow_html=True,
    )

    clean_a, clean_b = st.columns(2)

    with clean_a:
        st.markdown("#### Structure")

        if st.button("Remove duplicate rows", use_container_width=True):
            before = len(st.session_state.working_df)
            st.session_state.working_df = st.session_state.working_df.drop_duplicates()
            after = len(st.session_state.working_df)
            log_action(f"Removed {before - after} duplicate rows.")
            st.success(f"Removed {before - after} duplicate rows.")
            st.rerun()

        columns_to_drop = st.multiselect("Drop selected columns", df.columns.tolist())
        if st.button("Drop columns", use_container_width=True, disabled=not columns_to_drop):
            st.session_state.working_df = st.session_state.working_df.drop(columns=columns_to_drop)
            log_action(f"Dropped columns: {', '.join(columns_to_drop)}.")
            st.success("Selected columns dropped.")
            st.rerun()

        if st.button("Standardise column names", use_container_width=True):
            st.session_state.working_df.columns = make_unique_columns(st.session_state.working_df.columns)
            log_action("Standardised column names.")
            st.success("Column names standardised.")
            st.rerun()

    with clean_b:
        st.markdown("#### Rename One Column")

        if len(df.columns) > 0:
            old_name = st.selectbox("Column to rename", df.columns.tolist())
            new_name = st.text_input("New column name", value=old_name)

            if st.button("Rename column", use_container_width=True):
                if new_name.strip() == "":
                    st.warning("Please enter a valid column name.")
                elif new_name in df.columns and new_name != old_name:
                    st.warning("That column name already exists.")
                else:
                    st.session_state.working_df = st.session_state.working_df.rename(
                        columns={old_name: new_name.strip()}
                    )
                    log_action(f"Renamed {old_name} to {new_name.strip()}.")
                    st.success("Column renamed.")
                    st.rerun()

    st.divider()

    missing_a, missing_b = st.columns(2)

    with missing_a:
        st.markdown("#### Missing Values")
        missing_cols = df.columns[df.isna().any()].tolist()
        chosen_missing_cols = st.multiselect("Columns with missing values", missing_cols)
        missing_strategy = st.selectbox(
            "Missing-value strategy",
            [
                "Drop rows with missing values",
                "Fill with mean",
                "Fill with median",
                "Fill with mode",
                "Fill with zero",
                "Fill with custom value",
            ],
        )
        custom_fill_value = None
        if missing_strategy == "Fill with custom value":
            custom_fill_value = st.text_input("Custom fill value", value="Unknown")

        if st.button(
            "Apply missing-value strategy",
            use_container_width=True,
            disabled=not chosen_missing_cols,
        ):
            incompatible_numeric = (
                missing_strategy in ["Fill with mean", "Fill with median"]
                and any(col not in numeric_cols for col in chosen_missing_cols)
            )

            if incompatible_numeric:
                st.warning("Mean and median filling can only be applied to numeric columns.")
            else:
                st.session_state.working_df = apply_missing_strategy(
                    st.session_state.working_df,
                    chosen_missing_cols,
                    missing_strategy,
                    custom_fill_value,
                )
                log_action(
                    f"Applied missing-value strategy '{missing_strategy}' to: "
                    f"{', '.join(chosen_missing_cols)}."
                )
                st.success("Missing-value strategy applied.")
                st.rerun()

    with missing_b:
        st.markdown("#### Data Type Conversion")
        conversion_cols = st.multiselect("Columns to convert", df.columns.tolist(), key="convert_cols")
        target_type = st.selectbox(
            "Target data type",
            ["Numeric", "Text", "Date", "Category"],
        )

        if st.button("Convert selected columns", use_container_width=True, disabled=not conversion_cols):
            updated = st.session_state.working_df.copy()

            for col in conversion_cols:
                if target_type == "Numeric":
                    updated[col] = pd.to_numeric(updated[col], errors="coerce")
                elif target_type == "Text":
                    updated[col] = updated[col].astype("string")
                elif target_type == "Date":
                    updated[col] = pd.to_datetime(updated[col], errors="coerce")
                elif target_type == "Category":
                    updated[col] = updated[col].astype("category")

            st.session_state.working_df = updated
            log_action(f"Converted {', '.join(conversion_cols)} to {target_type}.")
            st.success("Data type conversion applied.")
            st.rerun()

    st.divider()

    outlier_a, text_b = st.columns(2)

    with outlier_a:
        st.markdown("#### Outliers")
        chosen_outlier_cols = st.multiselect("Numeric columns", numeric_cols, key="outlier_cols")
        outlier_action = st.radio(
            "Outlier treatment",
            ["Cap using IQR bounds", "Remove rows outside IQR bounds"],
            horizontal=False,
        )

        if st.button("Apply outlier treatment", use_container_width=True, disabled=not chosen_outlier_cols):
            before = len(st.session_state.working_df)

            if outlier_action == "Cap using IQR bounds":
                st.session_state.working_df = cap_outliers_iqr(
                    st.session_state.working_df,
                    chosen_outlier_cols,
                )
                log_action(f"Capped outliers using IQR for: {', '.join(chosen_outlier_cols)}.")
                st.success("Outliers capped.")
            else:
                st.session_state.working_df = remove_outliers_iqr(
                    st.session_state.working_df,
                    chosen_outlier_cols,
                )
                after = len(st.session_state.working_df)
                log_action(f"Removed {before - after} rows outside IQR bounds.")
                st.success(f"Removed {before - after} rows outside IQR bounds.")

            st.rerun()

    with text_b:
        st.markdown("#### Text Cleaning and Encoding")
        text_clean_cols = st.multiselect("Text columns to clean", categorical_cols, key="text_clean_cols")
        lower_case = st.checkbox("Convert text to lowercase", value=True)
        strip_spaces = st.checkbox("Trim extra spaces", value=True)

        if st.button("Clean selected text columns", use_container_width=True, disabled=not text_clean_cols):
            updated = st.session_state.working_df.copy()

            for col in text_clean_cols:
                updated[col] = updated[col].astype("string")
                if strip_spaces:
                    updated[col] = updated[col].str.strip().str.replace(r"\s+", " ", regex=True)
                if lower_case:
                    updated[col] = updated[col].str.lower()

            st.session_state.working_df = updated
            log_action(f"Cleaned text columns: {', '.join(text_clean_cols)}.")
            st.success("Text columns cleaned.")
            st.rerun()

        encode_cols = st.multiselect("Categorical columns to one-hot encode", categorical_cols)
        if st.button("One-hot encode selected columns", use_container_width=True, disabled=not encode_cols):
            st.session_state.working_df = pd.get_dummies(
                st.session_state.working_df,
                columns=encode_cols,
                drop_first=False,
                dtype=int,
            )
            log_action(f"One-hot encoded: {', '.join(encode_cols)}.")
            st.success("Categorical columns encoded.")
            st.rerun()


with eda_tab:
    st.subheader("Visual Exploration")

    chart_a, chart_b = st.columns(2)

    with chart_a:
        st.markdown("#### Numeric Distribution")
        if numeric_cols:
            hist_col = st.selectbox("Choose numeric column", numeric_cols, key="hist_col")
            fig = px.histogram(
                df,
                x=hist_col,
                marginal="box",
                color_discrete_sequence=["#2563eb"],
                title=f"Distribution of {hist_col}",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available.")

    with chart_b:
        st.markdown("#### Categorical Breakdown")
        if categorical_cols:
            cat_col = st.selectbox("Choose categorical column", categorical_cols, key="cat_col")
            counts = df[cat_col].astype(str).value_counts(dropna=False).head(20).reset_index()
            counts.columns = [cat_col, "count"]
            fig = px.bar(
                counts,
                x=cat_col,
                y="count",
                color="count",
                color_continuous_scale="Tealgrn",
                title=f"Top categories in {cat_col}",
            )
            fig.update_layout(template="plotly_white", xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns available.")

    chart_c, chart_d = st.columns(2)

    with chart_c:
        st.markdown("#### Relationship Explorer")
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y-axis", numeric_cols, index=1, key="scatter_y")
            color_col_options = ["None"] + categorical_cols
            color_col = st.selectbox("Optional color column", color_col_options)
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=None if color_col == "None" else color_col,
                title=f"{y_col} vs {x_col}",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("At least two numeric columns are needed for a scatterplot.")

    with chart_d:
        st.markdown("#### Boxplot")
        if numeric_cols:
            box_col = st.selectbox("Boxplot column", numeric_cols, key="box_col")
            group_options = ["None"] + categorical_cols
            group_col = st.selectbox("Optional grouping column", group_options)
            fig = px.box(
                df,
                x=None if group_col == "None" else group_col,
                y=box_col,
                color=None if group_col == "None" else group_col,
                title=f"Boxplot of {box_col}",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available.")

    st.markdown("#### Correlation Heatmap")
    if len(numeric_cols) >= 2:
        corr_cols = st.multiselect(
            "Select numeric columns for correlation",
            numeric_cols,
            default=numeric_cols[: min(8, len(numeric_cols))],
        )

        if len(corr_cols) >= 2:
            corr = df[corr_cols].corr(numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation Heatmap",
            )
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least two numeric columns.")
    else:
        st.info("At least two numeric columns are needed for a correlation heatmap.")


with export_tab:
    st.subheader("Review and Export")

    st.markdown("#### Cleaning History")
    if st.session_state.cleaning_log:
        for idx, item in enumerate(st.session_state.cleaning_log, start=1):
            st.write(f"{idx}. {item}")
    else:
        st.info("No cleaning actions have been applied yet.")

    st.markdown("#### Cleaned Dataset Preview")
    st.dataframe(df.head(200), use_container_width=True)

    export_a, export_b = st.columns(2)
    with export_a:
        st.download_button(
            "Download CSV",
            data=convert_df_to_csv(df),
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_b:
        st.download_button(
            "Download Excel",
            data=convert_df_to_excel(df),
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
