import os
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# Initialize the FastMCP server
mcp = FastMCP("HealthInsightsServer")

# Initialize the Gemini Client
# It will automatically pick up the GEMINI_API_KEY from the environment
gemini_client = genai.Client()

# Get the Laravel API base URL from the environment, with a fallback
LARAVEL_API_URL = os.getenv("LARAVEL_API_URL", "http://localhost:8000/api")

@mcp.tool()
async def get_health_insights(user_token: str) -> str:
    """
    Fetches the health data for the authenticated user from the Laravel API
    and uses Google Gemini to generate personalized health insights.
    
    Args:
        user_token: The Bearer token for the logged-in user.
    """
    # 1. Fetch user health data from Laravel
    headers = {
        "Authorization": f"Bearer {user_token}",
        "Accept": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            # Assumes your Laravel app has an endpoint returning the user's health JSON
            response = await client.get(f"{LARAVEL_API_URL}/user/health", headers=headers)
            response.raise_for_status() # Raise an error if status code is not 2xx
            health_data = response.json()
        except Exception as e:
            return f"Error fetching health data from Laravel: {str(e)}"
    
    # 2. Construct the prompt for Gemini
    prompt = f"""
    You are an expert health and wellness assistant. 
    Analyze the following user health data and provide brief, actionable insights.
    
    User Data:
    {health_data}
    
    Format the response clearly using Markdown. Focus on trends and actionable advice.
    """
    
    # 3. Call Gemini 1.5 Flash to generate insights
    try:
        gemini_response = gemini_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7, # Controls creativity. 0.7 gives a good balance.
            ),
        )
        return gemini_response.text
    except Exception as e:
         return f"Error generating insights with Gemini: {str(e)}"

if __name__ == "__main__":
    # Start the MCP server using standard input/output (stdio)
    # The client runs this script as a subprocess and talks to it over stdin/stdout.
    mcp.run()
