# ============================================================
# IMPORTS
# ============================================================

# StructuredTool allows us to convert a normal Python function
# into a LangChain Tool with a structured input schema.
from langchain_core.tools import StructuredTool

# Pydantic is used to define and validate the input that
# our LangChain Tool will receive.
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator
)


# ============================================================
# PYDANTIC INPUT MODEL
# ============================================================

# This model defines the structure and validation rules
# for the input received by our tool.
#
# The tool expects two inputs:
#     a -> integer
#     b -> integer
#
# Both values will be validated before the actual function
# is executed.

class MultiplyInput(BaseModel):

    # `...` means the field is required.
    #
    # Field() also allows us to provide metadata such as
    # descriptions that can be used by LangChain/LLMs.
    a: int = Field(
        ...,
        description="First number. Must be an integer between 1 and 100."
    )

    b: int = Field(
        ...,
        description="Second number. Must be an integer between 1 and 100."
    )


    # ========================================================
    # FIELD-LEVEL VALIDATION
    # ========================================================

    # `field_validator()` is used when we want to validate
    # individual fields.
    #
    # "a", "b" means this validator will run for both fields.
    #
    # The validator checks:
    #     1. Value must be an integer
    #     2. Value must be greater than 0
    #     3. Value must be less than or equal to 100

    @field_validator("a", "b")
    @classmethod
    def validate_numbers(cls, value):

        # Check whether the value is an integer.
        if not isinstance(value, int):
            raise ValueError(
                "Value must be an integer"
            )

        # Value must be greater than 0.
        if value <= 0:
            raise ValueError(
                "Value must be greater than 0"
            )

        # Value cannot be greater than 100.
        if value > 100:
            raise ValueError(
                "Value must be less than or equal to 100"
            )

        # Return the validated value.
        return value


    # ========================================================
    # MODEL-LEVEL VALIDATION
    # ========================================================

    # `model_validator()` is used when validation depends
    # on multiple fields together.
    #
    # Here we are validating the relationship between
    # `a` and `b`.

    @model_validator(mode="after")
    def validate_combination(self):

        # a and b cannot contain the same value.
        if self.a == self.b:
            raise ValueError(
                "a and b must be different numbers"
            )

        # Make sure the multiplication result does not
        # exceed 5000.
        if self.a * self.b > 5000:
            raise ValueError(
                "The multiplication result cannot be greater than 5000"
            )

        # Return the validated model.
        return self


# ============================================================
# NORMAL PYTHON FUNCTION
# ============================================================

# This is initially just a normal Python function.
#
# It accepts two integer arguments and returns their sum.

def addition(a: int, b: int) -> int:
    """
    Add two numbers and return their sum.
    """

    return a + b


# ============================================================
# CONVERT FUNCTION INTO LANGCHAIN STRUCTURED TOOL
# ============================================================

# `StructuredTool.from_function()` converts our normal
# Python function into a LangChain StructuredTool.
#
# This allows the function to be used by:
#     - LangChain Agents
#     - LLMs
#     - Tool calling
#
# `args_schema=MultiplyInput` tells LangChain that the
# input must follow the Pydantic validation rules defined
# above.

addition = StructuredTool.from_function(
    func=addition,

    # Name that will be exposed to the LLM/Agent.
    name="Addition Function",

    # Description tells the LLM what this tool does.
    description="Perform Addition over two provided int inputs",

    # Pydantic model that defines and validates the
    # tool's input parameters.
    args_schema=MultiplyInput
)


# ============================================================
# INVOKE THE STRUCTURED TOOL
# ============================================================

# `.invoke()` executes the LangChain tool.
#
# The dictionary keys must match the fields defined
# inside MultiplyInput.
#
# Before `addition()` executes, Pydantic validation
# will be performed.

result = addition.invoke({
    "a": 10,
    "b": 20
})


# ============================================================
# PRINT RESULT
# ============================================================

print("Addition Result:", result)