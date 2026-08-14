# ============================================================
# IMPORTS
# ============================================================

# BaseTool is the base class used to create custom LangChain tools.
#
# By inheriting from BaseTool, our MultipleTool gets the standard
# LangChain tool interface, including:
#
#   - invoke()
#   - run()
#   - ainvoke()
#   - arun()
#   - input validation
#   - tool metadata such as name and description
#
from langchain.tools import BaseTool


# Type is used for type annotations.
#
# We will use Type[BaseModel] below to tell Python that
# args_schema should contain a Pydantic model class.
#
from typing import Type


# Pydantic is used to define and validate the input that
# our tool receives.
#
# BaseModel       -> creates the input schema
# Field           -> adds descriptions/default/metadata
# field_validator -> validates individual fields
# model_validator -> validates the complete model
#
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator
)


# ============================================================
# 1. TOOL INPUT SCHEMA
# ============================================================

# This Pydantic model defines the INPUT expected by our tool.
#
# Our tool expects:
#
# {
#     "a": 10,
#     "b": 20
# }
#
# Both "a" and "b" must satisfy the validation rules below.
#
class MultiplyInput(BaseModel):

    # --------------------------------------------------------
    # First input parameter
    # --------------------------------------------------------

    a: int = Field(
        ...,
        description="First number. Must be an integer between 1 and 100."
    )

    # --------------------------------------------------------
    # Second input parameter
    # --------------------------------------------------------

    b: int = Field(
        ...,
        description="Second number. Must be an integer between 1 and 100."
    )


    # ========================================================
    # 2. FIELD VALIDATOR
    # ========================================================

    # @field_validator is used when we want to validate
    # individual fields.
    #
    # "a", "b" means this validator will run for BOTH fields.
    #
    @field_validator("a", "b")
    @classmethod
    def validate_numbers(cls, value):

        # Check whether the value is an integer.
        #
        # Pydantic already performs type validation for
        # "a: int" and "b: int", but we are explicitly
        # demonstrating the validation here.
        #
        if not isinstance(value, int):
            raise ValueError(
                "Value must be an integer"
            )


        # The number must be greater than 0.
        #
        # Therefore:
        #
        # 0  -> invalid
        # -5 -> invalid
        # 10 -> valid
        #
        if value <= 0:
            raise ValueError(
                "Value must be greater than 0"
            )


        # The number cannot be greater than 100.
        #
        # Therefore:
        #
        # 101 -> invalid
        # 100 -> valid
        # 50  -> valid
        #
        if value > 100:
            raise ValueError(
                "Value must be less than or equal to 100"
            )


        # If all validations pass, return the value.
        return value


    # ========================================================
    # 3. MODEL VALIDATOR
    # ========================================================

    # @model_validator validates the COMPLETE Pydantic model.
    #
    # This is different from @field_validator.
    #
    # field_validator:
    #     validates one field at a time.
    #
    # model_validator:
    #     validates the relationship between multiple fields.
    #
    # Here we need to compare "a" and "b", so model_validator
    # is appropriate.
    #
    @model_validator(mode="after")
    def validate_combination(self):

        # ----------------------------------------------------
        # Rule 1:
        # a and b cannot be the same number.
        # ----------------------------------------------------
        #
        # Example:
        #
        # a = 10
        # b = 10
        #
        # This should fail.
        #
        if self.a == self.b:
            raise ValueError(
                "a and b must be different numbers"
            )


        # ----------------------------------------------------
        # Rule 2:
        # Multiplication result cannot exceed 5000.
        # ----------------------------------------------------
        #
        # Example:
        #
        # a = 50
        # b = 100
        #
        # 50 * 100 = 5000
        #
        # This is valid.
        #
        # But:
        #
        # a = 60
        # b = 100
        #
        # 60 * 100 = 6000
        #
        # This is invalid.
        #
        if self.a * self.b > 5000:
            raise ValueError(
                "The multiplication result cannot be greater than 5000"
            )


        # Return the validated model.
        return self


# ============================================================
# 4. CUSTOM LANGCHAIN TOOL
# ============================================================

# We create our own tool by inheriting from BaseTool.
#
# BaseTool provides the standard LangChain Tool interface.
#
# Therefore:
#
#             BaseTool
#                 ↑
#                 |
#           MultipleTool
#
class MultipleTool(BaseTool):


    # --------------------------------------------------------
    # Tool Name
    # --------------------------------------------------------
    #
    # This is the name LangChain/Agent can use to identify
    # this tool.
    #
    name: str = "Muliply Tool"


    # --------------------------------------------------------
    # Tool Description
    # --------------------------------------------------------
    #
    # The description tells the LLM/Agent what this tool does.
    #
    # This becomes especially important when an Agent has
    # multiple tools and needs to decide which tool to call.
    #
    description: str = "This Tool is used for the Multiplication"


    # --------------------------------------------------------
    # Input Schema
    # --------------------------------------------------------
    #
    # args_schema tells LangChain:
    #
    # "What input does this tool expect?"
    #
    # Here we are saying:
    #
    # MultipleTool
    #      |
    #      ↓
    # MultiplyInput
    #
    # Therefore the tool expects:
    #
    # {
    #     "a": ...,
    #     "b": ...
    # }
    #
    args_schema: Type[BaseModel] = MultiplyInput


    # ========================================================
    # 5. TOOL EXECUTION LOGIC
    # ========================================================

    # _run() contains the ACTUAL work performed by the tool.
    #
    # BaseTool handles the tool execution process and eventually
    # calls this method.
    #
    # a and b come from the validated MultiplyInput schema.
    #
    def _run(self, a: int, b: int) -> int:

        # Perform multiplication and return the result.
        return a * b


# ============================================================
# 6. CREATE TOOL INSTANCE
# ============================================================

# MultipleTool is a class.
#
# We create an instance of that class so that we can execute it.
#
newTool = MultipleTool()


# ============================================================
# 7. INVOKE THE TOOL
# ============================================================

# invoke() is the standard Runnable interface method.
#
# IMPORTANT:
#
# Do NOT do:
#
#     newTool.invoke(10, 20)
#
# because invoke() expects:
#
#     invoke(input, config=None, ...)
#
# Therefore "20" would be interpreted as the config argument.
#
# Instead, because our tool has TWO named inputs defined by
# MultiplyInput, we pass a dictionary:
#
# {
#     "a": 10,
#     "b": 20
# }
#
print(
    newTool.invoke({
        "a": 10,
        "b": 20
    })
)