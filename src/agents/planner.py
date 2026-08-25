from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
import json
import re

class PlannerAgent:
    def __init__(self, model="openai/gpt-oss-120b"):  # 🔥 Changed to available model
        self.llm = ChatGroq(
            model=model,
            temperature=0.3,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def create_plan(self, df_head, df_info, df_shape, target_column):
        prompt = f"""
        You are a Senior Data Scientist. You need to create a detailed plan for analyzing the given dataset.

        Dataset Information:
        - Shape: {df_shape}
        - Columns: {df_info}
        - First 5 rows: {df_head}
        - Target Column: {target_column}

        Create a step-by-step plan for:
        1. Data Cleaning & Preprocessing (missing values, outliers, encoding)
        2. Exploratory Data Analysis (distributions, correlations, visualizations)
        3. Feature Engineering (feature creation, scaling, transformations)
        4. Model Selection & Training (at least 3 different models)
        5. Model Evaluation & Comparison
        6. Best Model Selection

        Output the plan as a JSON list of steps, where each step has:
        - "step": description of the step
        - "code": Python code to execute for this step
        - "expected_output": what this step will produce

        The code should use pandas, numpy, matplotlib, seaborn, sklearn, and xgboost.
        """
        
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        try:
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                plan = [
                    {"step": "Load and inspect data", "code": "df.head()", "expected_output": "Data preview"},
                    {"step": "Check missing values", "code": "df.isnull().sum()", "expected_output": "Missing values count"},
                    {"step": "Train Random Forest", "code": "from sklearn.ensemble import RandomForestRegressor; model = RandomForestRegressor(); model.fit(X_train, y_train)", "expected_output": "Trained model"},
                    {"step": "Evaluate model", "code": "from sklearn.metrics import r2_score; r2_score(y_test, y_pred)", "expected_output": "R2 score"}
                ]
        except:
            plan = [
                {"step": "Load and inspect data", "code": "df.head()", "expected_output": "Data preview"},
                {"step": "Check missing values", "code": "df.isnull().sum()", "expected_output": "Missing values count"},
                {"step": "Train Random Forest", "code": "from sklearn.ensemble import RandomForestRegressor; model = RandomForestRegressor(); model.fit(X_train, y_train)", "expected_output": "Trained model"},
                {"step": "Evaluate model", "code": "from sklearn.metrics import r2_score; r2_score(y_test, y_pred)", "expected_output": "R2 score"}
            ]
        
        return plan