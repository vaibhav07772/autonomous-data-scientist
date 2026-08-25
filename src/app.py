import streamlit as st
import pandas as pd
import os
import sys
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.supervisor import SupervisorAgent
from state import AgentState

st.set_page_config(page_title="🤖 Autonomous Data Scientist", layout="wide")

st.title("🤖 Autonomous Data Scientist")
st.markdown("*Upload a CSV, and the AI Agent will do EDA, Feature Engineering, Model Training, and Deployment Code Generation!*")

with st.sidebar:
    st.header("⚙️ Configuration")
    target_col = st.text_input("Target Column Name", "target")
    st.markdown("---")
    st.markdown("### 🧠 Agent Team")
    st.markdown("📋 **Planner** - Creates analysis plan")
    st.markdown("💻 **Coder** - Writes Python code")
    st.markdown("📊 **Analyst** - Interprets results")
    st.markdown("---")
    st.caption("Built with LangGraph + Groq + Streamlit")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        csv_path = tmp.name
    
    df = pd.read_csv(csv_path)
    st.write("### 📄 Data Preview")
    st.dataframe(df.head(10))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")
    
    if st.button("🚀 Run Autonomous Data Scientist", type="primary"):
        # Only numeric columns for correlation
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().to_string()
        else:
            corr_matrix = "No numeric columns for correlation"
        
        state = AgentState(
            csv_path=csv_path,
            target_column=target_col,
            df_head=df.head().to_string(),
            df_info=str(df.info()),
            df_shape=str(df.shape),
            missing_values=df.isnull().sum().to_dict(),
            column_types=df.dtypes.astype(str).to_dict(),
            correlation_matrix=corr_matrix,
            plan=[],
            current_step=0,
            generated_code="",
            execution_result=None,
            error=None,
            eda_summary="",
            model_results={},
            best_model="",
            best_score=0.0,
            final_report="",
            deployment_code="",
            iteration=0,
            max_iterations=1
        )
        
        with st.spinner("🧠 Agents are working..."):
            supervisor = SupervisorAgent()
            result = supervisor.run(state)
        
        st.success("✅ Analysis Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 EDA Summary")
            st.write(result.get("eda_summary", "No summary available"))
        with col2:
            st.subheader("🏆 Best Model")
            models = []
            for res in result.get("model_results", []):
                if "model" in str(res):
                    models.append(res)
            if models:
                st.write(f"Found {len(models)} trained models")
            else:
                st.write("Models trained successfully")
        
        st.subheader("📝 Final Report")
        st.write(result.get("final_report", "No report available"))
        
        with st.expander("🔍 Detailed Analysis Steps"):
            for i, analysis in enumerate(result.get("eda_summary", [])):
                st.write(f"**Step {i+1}:** {analysis}")
        
        with st.expander("📦 Deployment Code"):
            st.code(result.get("deployment_code", "# Code will be generated"), language="python")