import logging
import os
from typing import Optional, Tuple, Any
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class MathTool(BaseTool):
    """A tool for making mathematical calculations."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema
    
    async def create(self, instance_id: Optional[str], **kwargs) -> str:
        """Create a tool instance.
        
        Args:
            instance_id: The instance id of the tool.
            
        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "", "reward": 0.0
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the tool.
        
        Args:
            num1: The first number.
            num2: The second number.
            operation: The operation to perform.
        
        Returns: tool_response, tool_reward_score, tool_metrics
        """
        if "num1" not in parameters or "num2" not in parameters or "operation" not in parameters:
            return "Invalid parameters", 0.0, {}
        
        num1 = parameters.get("num1")
        num2 = parameters.get("num2")
        operation = parameters.get("operation")
        
        if not isinstance(num1, float) or not isinstance(num2, float):
            return "Invalid numbers", 0.0, {}
        if operation not in ["add", "subtract", "multiply", "divide"]:
            return "Invalid operation", 0.0, {}
        
        logger.info(f"Executing math tool with num1: {num1}, num2: {num2}, operation: {operation}")
        
        if operation == "add":
            result = num1 + num2
        elif operation == "subtract":
            result = num1 - num2
        elif operation == "multiply":
            result = num1 * num2
        elif operation == "divide":
            result = num1 / num2
        return result, 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.
        
        Args:
            instance_id: The instance id of the tool.
        """
        del self._instance_dict[instance_id]