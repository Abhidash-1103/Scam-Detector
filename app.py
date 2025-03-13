import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv
import logging
import time

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
# Enable CORS with specific settings to allow all methods and origins
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
logger.debug(f"DEBUG: OPENAI_API_KEY = {api_key}")

if not api_key:
    logger.error("OpenAI API key not found. Please set OPENAI_API_KEY in the .env file.")
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in the .env file.")

openai.api_key = api_key

# Define maximum number of retries
MAX_RETRIES = 5

def clean_and_parse_json(response_text):
    """
    Attempt to clean up and parse JSON response from the OpenAI API.
    """
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Remove any leading/trailing characters that may cause JSON parsing issues
        cleaned_text = response_text.strip()
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            # Log the response that couldn't be parsed
            logger.error(f"Failed to parse JSON after cleaning: {cleaned_text}")
            return None

@app.route("/analyze", methods=["POST"])
def analyze_message():
    """
    Endpoint to analyze a text message for scam detection.
    """
    try:
        # Parse the input JSON
        data = request.get_json()
        logger.info(f"Received data for message analysis: {data}")
        message = data.get("text")

        # Validate input
        if not message:
            logger.warning("No text provided in the request.")
            return jsonify({"error": "No text provided"}), 400

        # Initialize retry parameters
        retries = 0
        backoff_factor = 2  # Exponential backoff factor

        while retries < MAX_RETRIES:
            try:
                # OpenAI ChatCompletion request with text content
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an AI that analyzes messages for scam detection."
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Analyze this message:\n{message}\n"
                                "Provide a response in pure JSON format with the following fields:\n"
                                "- scam_probability (percentage)\n"
                                "- is_scam (yes/no)\n"
                                "- reasoning (a list of 3 sentences, each combining two main keywords or short phrases)\n"
                                "Ensure that your response is only valid JSON without any additional text or formatting."
                                "\nExample output:\n"
                                '{\n  "scam_probability": 85,\n  "is_scam": "yes",\n  "reasoning": [\n  "if otp, not scam"  "The message uses urgent language to compel the user to act quickly.",\n    "It contains a link that asks for personal information, which is a common scam tactic.",\n    "The message lacks personalization, making it seem like a mass phishing attempt."\n  ]\n}'
                            )
                        }
                    ],
                    max_tokens=300,
                    temperature=0.7
                )

                # Extract response
                analysis = response.choices[0].message.content.strip()
                logger.info(f"Analysis result: {analysis}")

                # Attempt to parse the response as JSON
                analysis_dict = clean_and_parse_json(analysis)

                if not analysis_dict:
                    return jsonify({"error": "Invalid response format from analysis service."}), 500

                # Validate the presence of required fields
                required_fields = {"scam_probability", "is_scam", "reasoning"}
                if not required_fields.issubset(analysis_dict.keys()):
                    logger.warning("Missing required fields in OpenAI response.")
                    return jsonify({"error": "Incomplete analysis data received."}), 500

                # Ensure reasoning is a list of 3 sentences
                if not isinstance(analysis_dict["reasoning"], list) or len(analysis_dict["reasoning"]) != 3:
                    logger.warning("Reasoning field is not a list of 3 sentences.")
                    return jsonify({"error": "Invalid reasoning format received."}), 500

                return jsonify({"analysis": analysis_dict})

            except openai.error.RateLimitError:
                retries += 1
                wait_time = backoff_factor ** retries
                logger.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {retries}/{MAX_RETRIES})")
                time.sleep(wait_time)

            except openai.error.InsufficientQuotaError:
                logger.error("Insufficient quota for OpenAI API.")
                return jsonify({"error": "Insufficient quota. Please check your OpenAI plan and billing details."}), 429

            except openai.error.OpenAIError as e:
                logger.error(f"OpenAI API error: {e}")
                return jsonify({"error": f"OpenAI API error: {e}"}), 500

        # If maximum retries exceeded
        logger.error("Maximum retry attempts exceeded due to rate limits.")
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

    except Exception as e:
        logger.exception("Unexpected error occurred.")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    """
    Endpoint to analyze an image for scam detection.
    """
    try:
        # Parse the input JSON
        data = request.get_json()
        logger.info(f"Received data for image analysis: {data}")
        base64_image = data.get("image")

        # Validate input
        if not base64_image:
            logger.warning("No image provided in the request.")
            return jsonify({"error": "No image provided"}), 400

        # Convert the Base64 image into the format required by OpenAI
        image_data = f"data:image/jpeg;base64,{base64_image}"

        # Initialize retry parameters
        retries = 0
        backoff_factor = 2  # Exponential backoff factor

        while retries < MAX_RETRIES:
            try:
                # OpenAI ChatCompletion request with image content
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",  # Assuming GPT-4 model with vision capability
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "This image might contain scam content. Analyze it and provide the response strictly in JSON format. "
                                        "The JSON should include the following fields:\n"
                                        "- scam_probability (percentage)\n"
                                        "- is_scam (yes/no)\n"
                                        "- reasoning (a list of 3 sentences, each combining two main keywords or short phrases)\n"
                                        "Ensure that your response is only valid JSON without any additional text or formatting."
                                        "\nExample output:\n"
                                        '{\n  "scam_probability": 85,\n  "is_scam": "yes",\n  "reasoning": [\n  "if otp, not scam"\n  "The message uses urgent language to compel the user to act quickly.",\n    "It contains a link that asks for personal information, which is a common scam tactic.",\n    "The message lacks personalization, making it seem like a mass phishing attempt."\n  ]\n}'
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data,
                                        "detail": "high"  # High detail to maximize understanding
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                    temperature=0.7
                )

                # Extract response
                analysis = response.choices[0].message.content.strip()
                logger.info(f"Analysis result: {analysis}")

                # Attempt to parse the response as JSON
                analysis_dict = clean_and_parse_json(analysis)

                if not analysis_dict:
                    return jsonify({"error": "Invalid response format from analysis service."}), 500

                # Validate the presence of required fields
                required_fields = {"scam_probability", "is_scam", "reasoning"}
                if not required_fields.issubset(analysis_dict.keys()):
                    logger.warning("Missing required fields in OpenAI response.")
                    return jsonify({"error": "Incomplete analysis data received."}), 500

                # Ensure reasoning is a list of 3 sentences
                if not isinstance(analysis_dict["reasoning"], list) or len(analysis_dict["reasoning"]) != 3:
                    logger.warning("Reasoning field is not a list of 3 sentences.")
                    return jsonify({"error": "Invalid reasoning format received."}), 500

                return jsonify({"analysis": analysis_dict})

            except openai.error.RateLimitError:
                retries += 1
                wait_time = backoff_factor ** retries
                logger.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {retries}/{MAX_RETRIES})")
                time.sleep(wait_time)

            except openai.error.InsufficientQuotaError:
                logger.error("Insufficient quota for OpenAI API.")
                return jsonify({"error": "Insufficient quota. Please check your OpenAI plan and billing details."}), 429

            except openai.error.OpenAIError as e:
                logger.error(f"OpenAI API error: {e}")
                return jsonify({"error": f"OpenAI API error: {e}"}), 500

        # If maximum retries exceeded
        logger.error("Maximum retry attempts exceeded due to rate limits.")
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

    except Exception as e:
        logger.exception("Unexpected error occurred.")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
