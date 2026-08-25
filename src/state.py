from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    # Input
    csv_path: str
    target_column: str
    
    # Data
    df_head: str
    df_info: str
    df_shape: str
    missing_values: Dict
    column_types: Dict
    correlation_matrix: str
    
    # Planning
    plan: List[str]
    current_step: int
    
    # Code
    generated_code: str
    execution_result: Any
    error: Optional[str]
    
    # Analysis
    eda_summary: str
    model_results: Dict[str, Any]
    best_model: str
    best_score: float
    
    # Final
    final_report: str
    deployment_code: str
    iteration: int
    max_iterations: int