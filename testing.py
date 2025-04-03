import os
import json
import base64
import subprocess
import time
import logging
import asyncio
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import openai
from elevenlabs import ElevenLabs
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Allow all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID", "9BWtsMINqrJLrRacOk9x")


# Hardcoded path for JSON data storage
User_messages=[]
JSON_FILE_PATH = "./userpersona.json"
EXCEL_FILE_PATH = "./User_Data.xlsx"
# Cache to avoid reprocessing same audio files
PROCESSED_FILES_CACHE = set()

class ChatRequest(BaseModel):
    message: Optional[str] = None

class UserPersonaUpdate(BaseModel):
    data: Dict[str, Any]

def exec_command(command: str) -> None:
    """Executes a shell command with proper error handling."""
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Command output: {process.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.cmd}")
        logger.error(f"Error output: {e.stderr}")
        raise RuntimeError(f"Command execution failed: {str(e)}")

async def lip_sync_message(message_index: int) -> None:
    """Convert the generated MP3 to WAV and create a lipsync JSON."""
    try:
        start_time = time.time()
        input_mp3 = f"audios/message_{message_index}.mp3"
        output_wav = f"audios/message_{message_index}.wav"
        output_json = f"audios/message_{message_index}.json"
        
        # Check if this file has already been processed
        cache_key = f"lipsync_{message_index}"
        if cache_key in PROCESSED_FILES_CACHE and os.path.exists(output_json):
            logger.info(f"Using cached lip sync data for message {message_index}")
            return
        
        # Ensure audios directory exists
        os.makedirs("audios", exist_ok=True)
        
        logger.info(f"Starting conversion for message {message_index}")
        
        # Using -y flag to overwrite existing files
        exec_command(f'ffmpeg -y -i {input_mp3} {output_wav}')
        logger.info(f"Conversion done in {int((time.time() - start_time) * 1000)}ms")
        
        # Change directory to rhubarb folder, execute the command, then change back
        current_dir = os.getcwd()
        rhubarb_dir = os.path.join(current_dir, "rhubarb")
        
        # Construct relative paths from rhubarb directory to the wav and json files
        relative_wav = os.path.join("..", output_wav)
        relative_json = os.path.join("..", output_json)
        
        os.chdir(rhubarb_dir)
        # Optimize Rhubarb by using phonetic recognition only (faster)
        # Add -q flag for quiet mode to reduce logging output
        exec_command(f'./rhubarb -q -f json -o {relative_json} {relative_wav} -r phonetic')
        os.chdir(current_dir)
        
        # Add to cache
        PROCESSED_FILES_CACHE.add(cache_key)
        
        logger.info(f"Lip sync done in {int((time.time() - start_time) * 1000)}ms")
    except Exception as e:
        logger.error(f"Error in lip sync processing: {str(e)}")
        raise

async def read_json_transcript(file_path: str) -> dict:
    """Reads a JSON file with proper error handling."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {str(e)}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error reading {file_path}: {str(e)}")
        return {}

async def audio_file_to_base64(file_path: str) -> str:
    """Converts audio file to base64 with proper error handling."""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error reading audio file {file_path}: {str(e)}")
        return ""

async def process_gpt_response(text: str, data_list: list[str]) -> List[Dict[str, Any]]:
    """Process text through GPT and return messages"""
    try:
        completion = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            max_tokens=1000,
            temperature=0.6,
            messages=[
                {
                    "role": "system",
                    "content": f"""You will always reply with a JSON array of messages. With a maximum of 1 messages.
        Each message has a text, facialExpression, and animation property. your replies always consist of 15 words not more not less you are always polite and you always keep the conversation engaging and going 
        The different facial expressions are: smile
        The different animations are: Talking_1 
        for your in the system prompt a history of the user messages will be provided so you have better conetet
        User_messages:{data_list} 
                    """
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )
        
        content_str = completion["choices"][0]["message"]["content"]
        messages_data = json.loads(content_str)
        
        if isinstance(messages_data, dict) and "messages" in messages_data:
            return messages_data["messages"]
        elif isinstance(messages_data, list):
            return messages_data
        else:
            raise ValueError("Unexpected format from OpenAI response")
    except Exception as e:
        logger.error(f"Error in GPT processing: {str(e)}")
        raise

async def generate_audio_responses(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate audio for each message using ElevenLabs"""
    try:
        client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)
        
        # Create a list of tasks to run in parallel
        lip_sync_tasks = []
        
        for i, message in enumerate(messages):
            text_input = message.get("text")
            if not text_input:
                continue

            file_name = f"audios/message_{i}.mp3"
            
            # Generate audio using ElevenLabs
            # Note: ElevenLabs currently only outputs MP3 directly, not WAV
            audio_bytes = b"".join(client.text_to_speech.convert(
                voice_id=VOICE_ID,
                output_format="mp3_44100_128",  # Highest quality MP3 for better conversions
                text=text_input,
                model_id="eleven_flash_v2_5", 
                optimize_streaming_latency=4,
            ))
            
            # Save the audio file
            with open(file_name, "wb") as f:
                f.write(audio_bytes)

            # Queue lip sync task to run in parallel
            lip_sync_tasks.append(lip_sync_message(i))

            # Add audio to message right away
            message["audio"] = await audio_file_to_base64(f"audios/message_{i}.mp3")
            
        # Run all lip sync tasks in parallel
        if lip_sync_tasks:
            await asyncio.gather(*lip_sync_tasks)
            
        # Now add lipsync data to messages after all processing is complete
        for i, message in enumerate(messages):
            if "text" in message:
                message["lipsync"] = await read_json_transcript(f"audios/message_{i}.json")
        
        return messages
    except Exception as e:
        logger.error(f"Error generating audio responses: {str(e)}")
        raise

async def call_chatgpt(prompt, model="gpt-4o-mini", api_key=None):
    """
    Call the ChatGPT API with a prompt.
    
    Args:
        prompt (str): Prompt to send to the API
        model (str): Model to use for the API call
        api_key (str): API key for OpenAI
        
    Returns:
        str: Response from the API
    """
    try:
        # If no API key is provided, use the one from the environment
        if api_key is None:
            api_key = openai.api_key
            
        # Using the OpenAI API to get a response
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content":  """You are a persona management assistant. Your job is to maintain a JSON object that tracks user information as it's shared in conversation. You'll receive two inputs:

The user's message
The current state of their persona JSON

When you receive new information in the user's message that isn't already in the JSON, update the JSON with that information. If fields are already populated, do not modify them. Add information that is explicitly mentioned or reasonably inferred from the user message, including inferring language proficiency based on the language the user communicates in.
IMPORTANT: Your output must be STRICTLY JSON format with no additional text, explanations, or commentary.
The persona JSON has these fields:

name: User's name
age: User's age (numeric value)
gender: User's gender identity
occupation: User's job or profession
location: User's city, country or region
interests: Array of the user's hobbies or interests
languages: Array of languages the user speaks (including languages inferred from their messages)
politicalIdeology: User's political leaning or beliefs
healthCondition: Any health conditions or concerns mentioned
behavioralPattern: Personality traits like "introvert", "extrovert", "ADHD", etc.
socialBehaviors: Deeper social tendencies, communication styles, or relationship patterns

Examples:
Example 1:
Current JSON:
{
  "name": "",
  "age": null,
  "gender": "",
  "occupation": "",
  "location": "",
  "interests": [],
  "languages": [],
  "politicalIdeology": "",
  "healthCondition": "",
  "behavioralPattern": "",
  "socialBehaviors": ""
}
User Message:
"Hi, I'm Sarah and I work as a software developer in Toronto."
Your Response (Complete JSON only):
{
  "name": "Sarah",
  "age": null,
  "gender": "",
  "occupation": "software developer",
  "location": "Toronto",
  "interests": [],
  "languages": ["English"],
  "politicalIdeology": "",
  "healthCondition": "",
  "behavioralPattern": "",
  "socialBehaviors": ""
}
Example 2:
Current JSON:
{
  "name": "",
  "age": null,
  "gender": "",
  "occupation": "",
  "location": "",
  "interests": [],
  "languages": [],
  "politicalIdeology": "",
  "healthCondition": "",
  "behavioralPattern": "",
  "socialBehaviors": ""
}
User Message:
"Hola, me llamo Carlos y soy estudiante de medicina en Barcelona."
Your Response (Complete JSON only):
{
  "name": "Carlos",
  "age": null,
  "gender": "male",
  "occupation": "medical student",
  "location": "Barcelona",
  "interests": [],
  "languages": ["Spanish"],
  "politicalIdeology": "",
  "healthCondition": "",
  "behavioralPattern": "",
  "socialBehaviors": ""
}
Example 3:
Current JSON:
{
  "name": "Michael",
  "age": 32,
  "gender": "male",
  "occupation": "",
  "location": "Berlin",
  "interests": ["photography"],
  "languages": ["German"],
  "politicalIdeology": "",
  "healthCondition": "",
  "behavioralPattern": "",
  "socialBehaviors": ""
}
User Message:
"I just got promoted to senior marketing manager last week! I've been dealing with anxiety for years but meditation helps. I usually need time alone after big social events to recharge."
Your Response (Complete JSON only):
{
  "name": "Michael",
  "age": 32,
  "gender": "male",
  "occupation": "senior marketing manager",
  "location": "Berlin",
  "interests": ["photography", "meditation"],
  "languages": ["German", "English"],
  "politicalIdeology": "",
  "healthCondition": "anxiety",
  "behavioralPattern": "introvert",
  "socialBehaviors": "needs alone time to recharge after socializing"
}
Example 4:
Current JSON:
{
  "name": "Akiko",
  "age": null,
  "gender": "female",
  "occupation": "",
  "location": "",
  "interests": [],
  "languages": [],
  "politicalIdeology": "",
  "healthCondition": "",
  "behavioralPattern": "",
  "socialBehaviors": ""
}
User Message:
"こんにちは、私は東京で美術教師をしています。絵を描くことと旅行が大好きです。"
Your Response (Complete JSON only):
{
  "name": "Akiko",
  "age": null,
  "gender": "female",
  "occupation": "art teacher",
  "location": "Tokyo",
  "interests": ["drawing", "traveling"],
  "languages": ["Japanese"],
  "politicalIdeology": "",
  "healthCondition": "",
  "behavioralPattern": "",
  "socialBehaviors": ""
}
Example 5:
Current JSON:
{
  "name": "Thomas",
  "age": 55,
  "gender": "male",
  "occupation": "retired",
  "location": "Florida",
  "interests": ["golf", "fishing"],
  "languages": [],
  "politicalIdeology": "",
  "healthCondition": "",
  "behavioralPattern": "",
  "socialBehaviors": ""
}
User Message:
"My diabetes has been acting up lately. I've been watching FOX News every night to keep up with what's happening. My wife says I should get out more, but I prefer small gatherings with close friends rather than big parties."
Your Response (Complete JSON only):
{
  "name": "Thomas",
  "age": 55,
  "gender": "male",
  "occupation": "retired",
  "location": "Florida",
  "interests": ["golf", "fishing", "watching news"],
  "languages": ["English"],
  "politicalIdeology": "likely conservative",
  "healthCondition": "diabetes",
  "behavioralPattern": "introvert",
  "socialBehaviors": "prefers intimate gatherings over large social events"
} one special case if you detect language urdu you have to mention it as hindi only not urdu and the entries you make in json should be in english only no other language  """},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error in call_chatgpt: {str(e)}")
        return f"An error occurred: {e}"

async def process_json_and_call_api(user_message):
    """
    Load JSON from a file, send it to the ChatGPT API with a user message,
    and update the original JSON file with the response.
    
    Args:
        user_message (str): User message (transcribed text) to send to the API
        
    Returns:
        dict: Updated JSON data with API response

    """
    logger.info(f"Processing JSON and calling API with user message: {user_message}")
    try:
        # Ensure the JSON file exists, create it if it doesn't
        if not os.path.exists(JSON_FILE_PATH):
            logger.info(f"Creating new JSON file at {JSON_FILE_PATH}")
            with open(JSON_FILE_PATH, 'w') as file:
                json.dump({"conversations": []}, file)
        logger.info(f"Reading JSON file from {JSON_FILE_PATH}")        
        # Load JSON from file
        with open(JSON_FILE_PATH, 'r') as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                # If the file exists but is empty or invalid, initialize it
                json_data = {"conversations": []}
        logger.info(f"Loaded JSON data: {json_data}")
        # Create prompt with JSON content and user message
        prompt = f"""Current JSON:
{json.dumps(json_data, indent=2)}

User Message: {user_message}"""
        
        # Call the API
        response = await call_chatgpt(prompt)
        
        # Parse the response and update the JSON file
        parsed_response = json.loads(response)
        
        # Write the updated JSON back to the file
        with open(JSON_FILE_PATH, 'w') as file:
            json.dump(parsed_response, file, indent=2)
            
        logger.info(f"Updated JSON file with new conversation entry")
        return json_data
        
    except Exception as e:
        logger.error(f"Error processing JSON and calling API: {e}")
        return None

@app.get("/")
async def root():
    return {"message": "Hello World!"}
@app.post("/update-user-persona")
async def update_user_persona(data: UserPersonaUpdate):
    try:
        logger.info(f"Updating user persona at {JSON_FILE_PATH}")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(JSON_FILE_PATH), exist_ok=True)
        
        # Write the updated data to the file
        with open(JSON_FILE_PATH, 'w') as file:
            json.dump(data.data, file, indent=2)
            
        logger.info(f"Successfully updated user persona")
        return {"success": True, "message": "User persona updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating user persona: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update user persona: {str(e)}")

@app.get("/get-user-persona")
async def get_user_persona():
    try:
        logger.info(f"Reading user persona from {JSON_FILE_PATH}")
        
        # Check if file exists
        if not os.path.exists(JSON_FILE_PATH):
            logger.info(f"User persona file not found, returning empty object")
            return {}
        
        # Read the file
        with open(JSON_FILE_PATH, 'r') as file:
            try:
                data = json.load(file)
                return data
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in user persona file")
                return {}
                
    except Exception as e:
        logger.error(f"Error reading user persona: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read user persona: {str(e)}")





# Path to save CSV data
def flatten_json(json_data, parent_key='', separator='_'):
    """Flatten a nested JSON structure into a single level dictionary."""
    items = {}
    for key, value in json_data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.update(flatten_json(value, new_key, separator))
        elif isinstance(value, list):
            # Convert list to string representation
            items[new_key] = json.dumps(value)
        else:
            items[new_key] = value
    return items

def save_json_to_excel(json_data, excel_file_path):
    flattened_data = flatten_json(json_data)
    
    # Create a DataFrame for the new row
    new_row_df = pd.DataFrame([flattened_data])
    
    # If the file exists, load existing data and append the new row
    if os.path.exists(excel_file_path):
        try:
            existing_df = pd.read_excel(excel_file_path)
            
            # Get all unique columns from both DataFrames
            all_columns = list(set(existing_df.columns).union(new_row_df.columns))
            
            # Reindex both DataFrames with the complete set of columns
            existing_df = existing_df.reindex(columns=all_columns)
            new_row_df = new_row_df.reindex(columns=all_columns)
            
            # For empty lists, use NA rather than [] to maintain consistency
            for col in new_row_df.columns:
                if isinstance(new_row_df[col].iloc[0], list) and len(new_row_df[col].iloc[0]) == 0:
                    new_row_df[col] = pd.NA
            
            # Append the new row to the existing DataFrame
            final_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        except Exception as e:
            print(f"Error reading existing Excel file: {e}")
            final_df = new_row_df
    else:
        # For new files, replace empty lists with NA
        for col in new_row_df.columns:
            if isinstance(new_row_df[col].iloc[0], list) and len(new_row_df[col].iloc[0]) == 0:
                new_row_df[col] = pd.NA
        final_df = new_row_df
    
    # Save the final DataFrame to Excel
    final_df.to_excel(excel_file_path, index=False)

def clear_json_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    """Clear all data but maintain structure."""
    if isinstance(data, dict):
        return {k: clear_json_structure(v) for k, v in data.items()}
    elif isinstance(data, list):
        return []
    elif isinstance(data, str):
        return ""
    elif isinstance(data, int):
        return 0
    elif isinstance(data, float):
        return 0.0
    elif isinstance(data, bool):
        return False
    return None

# Assume EXCEL_FILE_PATH is defined somewhere in your configuration.


@app.post("/end-chat")
async def end_chat(request: Request):
    # Get data from request
    data = await request.json()
    json_file_path = JSON_FILE_PATH
    
    if not json_file_path or not os.path.exists(json_file_path):
        return {"status": "error", "message": "Invalid or missing JSON file path"}
    
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            chat_data = json.load(file)
        
        # Update end time
        chat_data["end_time"] = data.get("endedAt", datetime.now().isoformat())
        
        # Save to Excel using our pandas-based function
        save_json_to_excel(chat_data, EXCEL_FILE_PATH)
        
        # Clear the JSON but maintain structure
        cleared_data = clear_json_structure(chat_data)
        User_messages.clear()
        
        # Write cleared data back to the JSON file
        with open(json_file_path, 'w') as file:
            json.dump(cleared_data, file, indent=2)
            
        return {"status": "success", "message": "Chat ended and data saved to Excel", "excel_path": EXCEL_FILE_PATH}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/voices")
async def get_voices():
    if not ELEVEN_LABS_API_KEY:
        raise HTTPException(status_code=400, detail="ELEVEN_LABS_API_KEY not set")
    client = ElevenLabs(api_key=ELEVEN_LABS_API_KEY)
    return client.voices.getAll()


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    
    try:
        logger.info("Starting audio transcription pipeline")
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            try:
                # Write the uploaded file content
                content = await audio.read()
                temp_audio.write(content)
                temp_audio.flush()
                
                # Step 1: Transcribe with Whisper
                with open(temp_audio.name, "rb") as audio_file:
                    logger.info("Transcribing audio with Whisper")
                    transcript = await openai.Audio.atranscribe(
                        "whisper-1",
                        audio_file
                    )
                    transcribed_text = transcript.text
                    logger.info(f"Transcription received: {transcribed_text}")
                
                # Start tasks that can run in parallel
                # Step 2: Process through GPT (this needs to finish before audio generation)
                logger.info("Processing transcription through GPT")
                User_messages.append(transcribed_text)
                messages_task = asyncio.create_task(process_gpt_response(transcribed_text , User_messages))
                
                # Step 3: Update JSON file (can run in parallel with everything else)
                logger.info("Updating JSON file with conversation data (parallel)")
                json_task = asyncio.create_task(process_json_and_call_api(transcribed_text))
                
                # Wait for GPT processing to complete
                messages = await messages_task
                
                # Step 4: Generate audio responses and lip sync (this will handle parallelization internally)
                logger.info("Generating audio responses and lip sync data")
                final_messages = await generate_audio_responses(messages)
                
                # We don't need to await the JSON task since it's running in parallel
                # and doesn't affect the response
                
                return {"messages": final_messages}
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_audio.name)
                except Exception as e:
                    logger.error(f"Error deleting temporary file: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error in transcribe pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)