from dotenv import load_dotenv
load_dotenv()

from src.agents.planner import PlannerAgent
from src.agents.coder import CoderAgent
from src.agents.analyst import AnalystAgent
from src.executor import CodeExecutor
from src.state import AgentState
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os

class SupervisorAgent:
    def __init__(self):
        self.planner = PlannerAgent()
        self.coder = CoderAgent()
        self.analyst = AnalystAgent()
        self.executor = CodeExecutor()
        self.plan = []
        self.results = []
        self.analysis = []

    def run(self, state: AgentState):
        print("🧠 Supervisor: Starting Autonomous Data Scientist...")
        df = pd.read_csv(state["csv_path"])
        print("📋 Planning...")
        self.plan = self.planner.create_plan(
            df.head().to_string(),
            str(df.info()),
            str(df.shape),
            state.get("target_column", "target")
        )
        for i, step in enumerate(self.plan):
            print(f"🔄 Executing step {i+1}/{len(self.plan)}: {step['step']}")
            code = self.coder.generate_code(
                step["step"],
                self.results,
                str(df.info()),
                state.get("target_column", "target")
            )
            result = self.executor.execute(code, df)
            self.results.append(result)
            analysis = self.analyst.analyze_results(
                step["step"],
                code,
                result,
                self.analysis
            )
            self.analysis.append(analysis)
        print("📊 Generating final report...")
        final_report = self.generate_final_report()
        state["eda_summary"] = self.analysis
        state["model_results"] = self.results
        state["final_report"] = final_report
        return state

    def generate_final_report(self):
        prompt = f"""
        You are a Data Scientist writing a final report. Based on the analysis results:

        Analysis steps completed: {len(self.analysis)}
        Results: {self.results}

        Write a comprehensive report covering:
        1. Data Overview (shape, columns, types)
        2. Data Quality (missing values, outliers, issues found)
        3. Key Insights (important patterns, correlations)
        4. Model Performance (best model, metrics)
        5. Recommendations for improvement

        Keep it professional and actionable.
        """
        llm = ChatGroq(
            model="openai/gpt-oss-120b", 
            temperature=0.2,
            api_key=os.getenv("GROQ_API_KEY")
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content