# agent_logic.py

import os
import json
import datetime
from dotenv import load_dotenv
from typing import Union, Dict
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain import hub
import requests

# Load environment variables from .env file
load_dotenv()

# --- TOOL DEFINITIONS ---

@tool
def web_search(query: str) -> str:
    """
    Searches the web for a single, specific query to get information using Google Search.
    Only use this for one search at a time. For example, you can search for
    'best beaches in Goa' or 'seafood restaurants in Calangute', but not both at once.
    """
    from langchain_community.utilities import GoogleSearchAPIWrapper
    
    print(f"--- Performing Google Search for: {query} ---")
    
    # This sets up the Google Search tool using the keys from your .env file
    search = GoogleSearchAPIWrapper(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        google_cse_id=os.getenv("GOOGLE_CSE_ID"),
    )
    
    # .run() executes the search
    return search.run(query)

@tool
def get_current_date(query: str = "") -> str:
    """
    Use this tool to get today's date to resolve any relative date queries like 'next month'.
    It takes no arguments.
    """
    return f"Today's date is {datetime.date.today().strftime('%A, %B %d, %Y')}."

@tool
def search_hotels(origin: str, destination: str = "", check_in_date: str = "", check_out_date: str = "") -> str:
    """
    Finds hotels for a given destination and dates.

    Args:
        origin (str): A JSON string containing the keys 'destination', 'check_in_date', and 'check_out_date'.
    """
    try:
        # This logic handles the case where the agent sends a JSON string in the first argument.
        input_dict = json.loads(origin)
        destination = input_dict['destination']
        check_in_date = input_dict['check_in_date']
        check_out_date = input_dict['check_out_date']

        print(f"--- Searching hotels in {destination} from {check_in_date} to {check_out_date} ---")
        return f"Found 3 hotels in {destination}: 1. The Grand Hotel (₹8000/night, 4.5 stars), 2. City Inn (₹4500/night, 4.0 stars), 3. Budget Stay (₹2500/night, 3.5 stars)."
    except Exception as e:
        return f"Error processing hotel search: {e}. Please ensure the input is a valid JSON with 'destination', 'check_in_date', and 'check_out_date' keys."

@tool
def search_flights(origin: str, destination: str = "", departure_date: str = "") -> str:
    """
    Finds real-time flights for a given origin, destination, and date using a specific flight data API.

    Args:
        origin (str): A JSON string containing the keys 'origin', 'destination', and 'departure_date'.
                      Example: {"origin": "Mumbai", "destination": "Goa", "departure_date": "2025-10-15"}
    """
    try:
        input_dict = json.loads(origin)
        origin_city = input_dict['origin']
        destination_city = input_dict['destination']
        date_val = input_dict['departure_date']

        print(f"--- Searching REAL flights from {origin_city} to {destination_city} for {date_val} ---")

        # --- IMPORTANT: UPDATE THESE VALUES ---
        # Replace these with the actual URL, parameters, and headers from your API provider.
        api_url = "https://google-flights2.p.rapidapi.com" # <-- UPDATE THIS
        querystring = {
            "departure_airport_code": "BOM", # Example for Mumbai
            "arrival_airport_code": "GOI",   # Example for Goa
            "date": date_val,
            "currency": "INR"
        }
        headers = {
            "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY"),
            "X-RapidAPI-Host": "google-flights2.p.rapidapi.com"
        }
        # ------------------------------------

        response = requests.get(api_url, headers=headers, params=querystring)
        response.raise_for_status() # This will raise an error for bad responses (4xx or 5xx)
        
        data = response.json()

        # --- NEW PARSING LOGIC FOR YOUR SPECIFIC API ---
        if not data or not data.get("data") or not data["data"].get("topFlights"):
            return f"No flights found from {origin_city} to {destination_city} on {date_val}."
            
        # Extract information from the 'topFlights' list
        formatted_flights = []
        for flight in data["data"]["topFlights"][:3]: # Get the top 3 results
            formatted_flights.append(
                f"- Airline: {flight.get('flights', [{}])[0].get('airline', 'N/A')}, "
                f"Price: ${flight.get('price', 'N/A')}, " # Assuming the price is in USD as per your example
                f"Duration: {flight.get('duration', {}).get('text', 'N/A')}, "
                f"Stops: {flight.get('stops', 'N/A')}"
            )
        
        if not formatted_flights:
             return f"Could not parse flight information from {origin_city} to {destination_city}."

        return "Here are the top flight options:\n" + "\n".join(formatted_flights)

    except requests.exceptions.RequestException as e:
        return f"API request failed: {e}"
    except Exception as e:
        return f"An error occurred while searching for flights: {e}. Please ensure the input is a valid JSON and the date is in YYYY-MM-DD format."

# List of all tools
tools = [web_search, search_hotels, search_flights, get_current_date]

# --- AGENT CREATION ---

def create_trip_planner_agent():
    """
    Initializes and returns the trip planning agent.
    """
    prompt = hub.pull("hwchase17/react")
    
    prompt.template = prompt.template.replace(
        "Action Input: the input to the action\n",
        "Action Input: for tools with multiple arguments, this MUST be a single line of JSON in the format {{\"arg_name\": \"value\"}}. For tools with a single string argument, this can be a simple string.\n"
    )

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )

    # We are NOT using llm.bind_tools() here to avoid the schema conflict.
    # We pass the plain LLM and the tools list directly to the agent.
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True
    )
    
    return agent_executor

# Create a single instance of the agent to be used by the API
trip_planner_agent = create_trip_planner_agent()