# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from agent_logic import trip_planner_agent
from typing import Optional

# Initialize the FastAPI app
app = FastAPI(
    title="Atlas Agent API",
    description="API for the AI-Powered Trip Planner Agent",
    version="1.0.0"
)

# --- NEW Structured UserRequest Model ---
# We now ask for specific details from the user.
class UserRequest(BaseModel):
    origin: str
    destination: str
    start_date: str
    duration_days: int
    notes: Optional[str] = None # For extra details like "I like history"

# --- NEW Endpoint Logic ---
@app.post("/plan-trip")
async def plan_trip(request: UserRequest):
    """
    Receives structured user input and returns a generated travel plan.
    """
    
    # Construct a detailed, structured prompt for the agent
    # This guides the agent and makes its job much easier.
    master_prompt = f"""
    You are an expert travel planner. Your task is to create a detailed itinerary based on the following information:
    - Origin: {request.origin}
    - Destination: {request.destination}
    - Start Date: {request.start_date}
    - Trip Duration: {request.duration_days} days

    User's additional notes: {request.notes if request.notes else "None"}

    Please perform the following steps:
    1. Search for round-trip flights for the given origin, destination, and start date.
    2. Based on the start date and duration, determine the check-in and check-out dates.
    3. Search for 3-4 highly-rated hotel options for these dates.
    4. Search for the top 3-5 points of interest or activities at the destination, keeping the user's notes in mind.
    5. Synthesize all this information into a complete, day-by-day itinerary.
    IMPORTANT: Your final response MUST start with the words "Final Answer:" and should contain ONLY the detailed itinerary that follows. Do not take any more actions after this.
    
    Your final answer must be only the detailed itinerary. It should include daily activities and a summary of the flight and hotel options you found.
    """
        
    agent_input = {"input": master_prompt}
    
    try:
        response = trip_planner_agent.invoke(agent_input)
        return {"plan": response['output']}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

@app.get("/")
def read_root():
    return {"status": "Atlas Agent is running!"}