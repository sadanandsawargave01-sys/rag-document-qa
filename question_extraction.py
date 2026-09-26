
from config import GOOGLE_API_KEY
from prompts import IMAGE_QUESTION_EXTRACTION_PROMPT

import google.generativeai as genai
import re
import json
from PIL import Image
import io


def detect_language(text: str) -> str:
    try:
        if re.search(r'[\u0900-\u097F]', text):
            if any(word in text for word in ["आहे", "म्हणजे", "तुम्ही", "काय"]):
                return "mr"
            return "hi"
        return "en"
    except:
        return "en"


def build_prompt(target_language: str = None) -> str:
    prompt = f"""
{IMAGE_QUESTION_EXTRACTION_PROMPT}

IMPORTANT:
- The image may contain Hindi, Marathi, or English text.
- Extract ALL questions exactly as they appear.
"""

    if target_language:
        prompt += f"""
- Translate all extracted questions into {target_language}.
"""
    else:
        prompt += """
- DO NOT translate the questions.
- Preserve original language.
"""

    prompt += """
Return STRICT JSON format only:
```json
{
  "queries": ["question1", "question2"]
}
"""
    return prompt

def image_question_extraction(image_bytes, target_language: str = None):
    try:
        genai.configure(api_key=GOOGLE_API_KEY)

        image_obj = Image.open(io.BytesIO(image_bytes))

        prompt = build_prompt(target_language)

        model = genai.GenerativeModel('gemini-2.5-flash')

        response = model.generate_content([image_obj, prompt])

        match = re.search(r'```(?:json)?\s*(.*?)\s*```', response.text, re.DOTALL)

        if match:
            json_str = match.group(1)
            data = json.loads(json_str)
            return data.get("queries", [])
        else:
            return []

    except Exception as e:
        raise Exception(f"Error in image_question_extraction: {e}")