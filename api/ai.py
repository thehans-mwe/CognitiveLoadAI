"""
CognitiveLoad AI - Groq AI Endpoint
====================================
AI-powered text simplification, study tips, and explanations
using Groq's ultra-fast LLM inference API.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import re
import time
import urllib.request
import urllib.error

# ============================================================================
# CONFIGURATION
# ============================================================================

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '').strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_PRIMARY = "llama-3.3-70b-versatile"
MODEL_FALLBACK = "llama-3.1-8b-instant"
MAX_INPUT_CHARS = 12000
GROQ_TIMEOUT = 25  # seconds

# ============================================================================
# PROMPTS
# ============================================================================

SIMPLIFY_SYSTEM = (
    "You are a world-class educational content simplifier. "
    "You transform complex academic text into clear, easy-to-understand language "
    "while preserving all key information and meaning."
)

SIMPLIFY_USER = """Rewrite the following text to be dramatically easier to understand.

Guidelines:
- Target a 7th-8th grade reading level
- Replace ALL jargon and complex vocabulary with everyday words
- Break long sentences into shorter ones (max 15-20 words each)
- Use active voice wherever possible
- Keep ALL facts, concepts, and key information intact
- Add brief parenthetical clarifications for any unavoidable technical terms
- Maintain the original paragraph structure

TEXT TO SIMPLIFY:
\"\"\"
{text}
\"\"\"

You MUST respond with valid JSON only (no markdown):
{{
  "simplified_text": "your fully simplified version of the text",
  "key_changes": ["description of change 1", "description of change 2", "description of change 3"],
  "difficulty_reduction": "one sentence summarizing how much easier the text is now"
}}"""

TIPS_SYSTEM = (
    "You are an expert learning coach and study strategist. "
    "You provide personalized, actionable study advice based on "
    "analysis of specific study materials."
)

TIPS_USER = """This study material has a cognitive load score of {score}/100 ({classification} difficulty).

STUDY MATERIAL:
\"\"\"
{text}
\"\"\"

Provide exactly 5 specific, actionable study tips for THIS exact material. Reference specific parts of the content. Be concrete and practical - no generic advice.

You MUST respond with valid JSON only (no markdown):
{{
  "tips": [
    {{"emoji": "📖", "title": "Short Title", "advice": "Detailed, actionable advice referencing the specific content"}},
    {{"emoji": "🧠", "title": "Short Title", "advice": "Detailed, actionable advice referencing the specific content"}},
    {{"emoji": "✍️", "title": "Short Title", "advice": "Detailed, actionable advice referencing the specific content"}},
    {{"emoji": "🔄", "title": "Short Title", "advice": "Detailed, actionable advice referencing the specific content"}},
    {{"emoji": "💡", "title": "Short Title", "advice": "Detailed, actionable advice referencing the specific content"}}
  ],
  "study_approach": "One paragraph describing the ideal overall study approach for this material"
}}"""

EXPLAIN_SYSTEM = (
    "You are a patient, brilliant tutor who explains complex concepts "
    "in simple terms using relatable analogies and real-world examples. "
    "You make learning feel effortless."
)

EXPLAIN_USER = """Explain the following text as if you're teaching a smart 14-year-old. Use analogies, everyday examples, and break down each concept in a fun, engaging way.

TEXT TO EXPLAIN:
\"\"\"
{text}
\"\"\"

You MUST respond with valid JSON only (no markdown):
{{
  "explanation": "Your clear, engaging explanation with analogies and examples",
  "key_concepts": [
    {{"term": "concept name", "simple_meaning": "simple one-line explanation"}},
    {{"term": "concept name", "simple_meaning": "simple one-line explanation"}}
  ],
  "analogy": "A relatable real-world analogy for the main idea of the text"
}}"""

# ============================================================================
# GROQ API CALLER
# ============================================================================

def _parse_json_safe(text):
    """Parse JSON from AI response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?\s*```$', '', text)
    return json.loads(text)


def _call_groq(messages, model=MODEL_PRIMARY, max_tokens=2048, temperature=0.7):
    """Call Groq API and return parsed JSON response."""
    if not GROQ_API_KEY:
        raise ValueError("AI service not configured — missing API key")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "CognitiveLoadAI/1.0"
    }

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"}
    }).encode('utf-8')

    req = urllib.request.Request(GROQ_URL, data=payload, headers=headers, method='POST')

    try:
        with urllib.request.urlopen(req, timeout=GROQ_TIMEOUT) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            content = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            parsed = _parse_json_safe(content)
            parsed['_ai_meta'] = {
                'model': model,
                'tokens_used': usage.get('total_tokens', 0),
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0)
            }
            return parsed
    except urllib.error.HTTPError as e:
        error_body = ''
        try:
            error_body = e.read().decode('utf-8')[:300]
        except Exception:
            pass
        if e.code == 429:
            raise ValueError("AI rate limit reached. Please wait a moment and try again.")
        if e.code == 401:
            raise ValueError("AI authentication failed.")
        if e.code == 503 or e.code == 502:
            raise ValueError("AI service temporarily unavailable. Please try again.")
        raise ValueError(f"AI service error ({e.code})")
    except urllib.error.URLError as e:
        raise ValueError(f"Could not reach AI service: {str(e.reason)}")
    except json.JSONDecodeError:
        raise ValueError("AI returned an invalid response. Please try again.")
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)[:100]}")


# ============================================================================
# MODE HANDLERS
# ============================================================================

def handle_simplify(text):
    """AI-powered text simplification using Groq LLM."""
    messages = [
        {"role": "system", "content": SIMPLIFY_SYSTEM},
        {"role": "user", "content": SIMPLIFY_USER.format(text=text[:MAX_INPUT_CHARS])}
    ]
    return _call_groq(messages, max_tokens=3000, temperature=0.5)


def handle_tips(text, score=50, classification="Medium"):
    """AI-powered personalized study tips."""
    messages = [
        {"role": "system", "content": TIPS_SYSTEM},
        {"role": "user", "content": TIPS_USER.format(
            text=text[:MAX_INPUT_CHARS],
            score=score,
            classification=classification
        )}
    ]
    return _call_groq(messages, max_tokens=1500, temperature=0.7)


def handle_explain(text):
    """AI-powered concept explanation with analogies."""
    messages = [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": EXPLAIN_USER.format(text=text[:MAX_INPUT_CHARS])}
    ]
    return _call_groq(messages, max_tokens=2000, temperature=0.6)


# ============================================================================
# VERCEL SERVERLESS HANDLER
# ============================================================================

class handler(BaseHTTPRequestHandler):
    def _send_json(self, status, data):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self._send_json(200, {'status': 'ok'})

    def do_GET(self):
        """Health check endpoint."""
        self._send_json(200, {
            'status': 'healthy',
            'service': 'CognitiveLoad AI — Groq',
            'version': '1.0.0',
            'models': [MODEL_PRIMARY, MODEL_FALLBACK],
            'modes': ['simplify', 'tips', 'explain'],
            'configured': bool(GROQ_API_KEY)
        })

    def do_POST(self):
        t_start = time.time()
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_json(400, {'error': 'Empty request body', 'code': 'EMPTY_BODY'})
                return
            if content_length > 100000:
                self._send_json(413, {'error': 'Request too large', 'code': 'TOO_LARGE'})
                return

            body = self.rfile.read(content_length).decode('utf-8')

            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send_json(400, {'error': 'Invalid JSON', 'code': 'INVALID_JSON'})
                return

            text = data.get('text', '').strip()
            mode = data.get('mode', 'simplify')

            if not text:
                self._send_json(400, {'error': 'No text provided', 'code': 'MISSING_TEXT'})
                return

            if len(text.split()) < 5:
                self._send_json(400, {
                    'error': 'Text too short. Please provide at least 5 words.',
                    'code': 'TEXT_TOO_SHORT'
                })
                return

            if not GROQ_API_KEY:
                self._send_json(503, {
                    'error': 'AI service not configured. Falling back to basic mode.',
                    'code': 'NO_API_KEY',
                    'fallback': True
                })
                return

            # Route to appropriate handler
            if mode == 'simplify':
                result = handle_simplify(text)
            elif mode == 'tips':
                score = data.get('score', 50)
                classification = data.get('classification', 'Medium')
                result = handle_tips(text, score, classification)
            elif mode == 'explain':
                result = handle_explain(text)
            else:
                self._send_json(400, {
                    'error': f'Unknown mode: {mode}. Use: simplify, tips, explain',
                    'code': 'INVALID_MODE'
                })
                return

            result['mode'] = mode
            result['processing_ms'] = round((time.time() - t_start) * 1000, 1)
            self._send_json(200, result)

        except ValueError as e:
            self._send_json(422, {
                'error': str(e),
                'code': 'AI_ERROR',
                'fallback': True,
                'processing_ms': round((time.time() - t_start) * 1000, 1)
            })
        except Exception as e:
            self._send_json(500, {
                'error': 'AI service error. Please try again.',
                'code': 'INTERNAL_ERROR',
                'fallback': True,
                'processing_ms': round((time.time() - t_start) * 1000, 1)
            })
