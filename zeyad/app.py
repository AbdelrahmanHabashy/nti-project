import streamlit as st
import tempfile
import os
import pandas as pd

from pipeline import process_invoice
from agent import graph_agent
from charts import create_chart


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Invoice AI Analytics",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Invoice AI Analytics")


# ==================================================
# Session State
# ==================================================

if "invoice_data" not in st.session_state:
    st.session_state.invoice_data = None


# ==================================================
# Invoice Upload
# ==================================================

st.header("📄 Upload Invoice")


uploaded_file = st.file_uploader(
    "Upload an invoice image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Invoice",
        width=500
    )

    if st.button("Extract Invoice Data"):

        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name

        try:

            # ==========================================
            # OCR + Invoice Extraction
            # ==========================================

            with st.spinner("Reading invoice..."):

                invoice_data = process_invoice(
                    temp_path
                )


            # ==========================================
            # Save extracted invoice
            # ==========================================

            st.session_state.invoice_data = invoice_data


            st.success(
                "Invoice processed successfully!"
            )


            # ==========================================
            # Display extracted data
            # ==========================================

            st.subheader(
                "Extracted Invoice Data"
            )

            st.json(invoice_data)


        except Exception as e:

            st.error(
                f"Error while processing invoice:\n{e}"
            )


        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)


# ==================================================
# Chart Agent
# ==================================================

st.divider()

st.header("📊 Invoice Analytics")


user_query = st.text_input(
    "What would you like to see?",
    placeholder=(
        "Example: Show total sales by product "
        "as a bar chart"
    )
)


if st.button("Generate Chart"):

    # ==========================================
    # Validation
    # ==========================================

    if not user_query.strip():

        st.warning(
            "Please enter a request."
        )


    elif st.session_state.invoice_data is None:

        st.warning(
            "Please upload and extract an invoice first."
        )


    else:

        # ==========================================
        # Get Invoice Data
        # ==========================================

        invoice_data = (
            st.session_state.invoice_data
        )


        # ==========================================
        # Convert Invoice to DataFrame
        # ==========================================

        df = pd.DataFrame(
            [invoice_data]
        )


        # ==========================================
        # Process Date
        # ==========================================

        if "date" in df.columns:

            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

            df["month"] = (
                df["date"]
                .dt
                .month_name()
            )


        # ==========================================
        # Ask Graph Agent
        # ==========================================

        try:

            with st.spinner(
                "AI is deciding the best chart..."
            ):

                spec = graph_agent(
                    user_query,
                    df
                )


            # ==========================================
            # Show AI Decision
            # ==========================================

            st.subheader(
                "AI Decision"
            )

            st.json(spec)


            # ==========================================
            # Create Chart
            # ==========================================

            fig = create_chart(
                df=df,
                chart_type=spec["chart_type"],
                x_column=spec["x_column"],
                y_column=spec["y_column"],
                title=spec["title"]
            )


            # ==========================================
            # Display Chart
            # ==========================================

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        except Exception as e:

            st.error(
                f"Error while generating chart:\n{e}"
            )