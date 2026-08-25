from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
import json
import re

class AnalystAgent:
    def __init__(self, model="openai/gpt-oss-120b"):  
        self.llm = ChatGroq(
            model=model,
            temperature=0.2,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def analyze_results(self, step, code, result, previous_analysis):
        prompt = f"""
        You are a Data Science Analyst. Analyze the results of the following step.

        Step: {step}
        Code executed: {code}
        Output: {result}
        Previous analysis: {previous_analysis}

        Provide:
        1. Summary of what this step achieved
        2. Key insights from the results
        3. Recommendations for the next step
        4. Any issues or concerns

        Return a JSON object with keys: "summary", "insights", "recommendations", "issues".
        """
        
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "summary": f"Completed step: {step}",
            "insights": "Results processed",
            "recommendations": "Proceed to next step",
            "issues": "None"
        }