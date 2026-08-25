from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os

class CoderAgent:
    def __init__(self, model="openai/gpt-oss-120b"):  
        self.llm = ChatGroq(
            model=model,
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def generate_code(self, plan_step, previous_results, df_info, target_column):
        prompt = f"""
        You are an expert Python developer. Write Python code to execute the following step.

        Step: {plan_step}
        Previous results: {previous_results}
        Dataset info: {df_info}
        Target column: {target_column}

        Rules:
        1. Use pandas as 'pd', numpy as 'np'
        2. Use matplotlib.pyplot as 'plt'
        3. Use seaborn as 'sns'
        4. The dataframe is already loaded as 'df'
        5. For XGBoost, use: from xgboost import XGBRegressor
        6. For splitting data, use: from sklearn.model_selection import train_test_split
        7. For metrics, use: from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        8. Print results using print()
        9. Store important results in variables
        10. Create visualizations and save them

        Return ONLY the Python code, no explanations.
        """
        
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content