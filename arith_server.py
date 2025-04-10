from typing import Any
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("arithmetic")

@mcp.tool()
async def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    """
    return a + b

@mcp.tool()
async def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first.
    
    Args:
        a: First number
        b: Second number
    """
    return a - b

@mcp.tool()
async def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
    """
    return a * b

@mcp.tool()
async def divide(a: float, b: float) -> float:
    """Divide the first number by the second.
    
    Args:
        a: First number (numerator)
        b: Second number (denominator)
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@mcp.tool()
async def power(base: float, exponent: float) -> float:
    """Raise a number to a power.
    
    Args:
        base: The base number
        exponent: The exponent
    """
    return base ** exponent

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')