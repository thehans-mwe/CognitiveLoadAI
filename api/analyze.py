"""
CognitiveLoad AI - Vercel Serverless API
=========================================
Analyzes educational text for cognitive load prediction.
"""

from http.server import BaseHTTPRequestHandler
import json
import re
import math
from collections import Counter
from typing import Dict, List, Tuple

# ============================================================================
# CONSTANTS
# ============================================================================

ABSTRACT_SUFFIXES = [
    'tion', 'sion', 'ness', 'ment', 'ity', 'ence', 'ance', 
    'ship', 'dom', 'hood', 'ism', 'ology', 'phy', 'ics'
]

TECHNICAL_PATTERNS = [
    r'\b\w+ization\b', r'\b\w+ification\b', r'\b\w+ological\b',
    r'\b\w+ometric\b', r'\b\w+aneous\b', r'\b\w+itious\b'
]

FEATURE_WEIGHTS = {
    'sentence_length': 0.20,
    'vocabulary_complexity': 0.25,
    'abstract_density': 0.20,
    'concept_repetition': -0.10,
    'readability': 0.25
}

LOAD_THRESHOLDS = {
    'low': 35,
    'medium': 65,
    'high': 100
}

# Common English stopwords
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

COMMON_WORDS = STOPWORDS.union({
    'make', 'made', 'take', 'took', 'give', 'gave', 'get', 'got',
    'know', 'knew', 'think', 'thought', 'see', 'saw', 'come', 'came',
    'want', 'use', 'find', 'found', 'tell', 'told', 'ask', 'work',
    'seem', 'feel', 'try', 'leave', 'call', 'good', 'new', 'first',
    'last', 'long', 'great', 'little', 'own', 'other', 'old', 'right',
    'big', 'high', 'different', 'small', 'large', 'next', 'early',
    'young', 'important', 'few', 'public', 'bad', 'same', 'able',
    'time', 'year', 'people', 'way', 'day', 'man', 'thing', 'woman',
    'life', 'child', 'world', 'school', 'state', 'family', 'student',
    'group', 'country', 'problem', 'hand', 'part', 'place', 'case',
    'week', 'company', 'system', 'program', 'question', 'work',
    'government', 'number', 'night', 'point', 'home', 'water', 'room',
    'mother', 'area', 'money', 'story', 'fact', 'month', 'lot', 'study',
    'book', 'eye', 'job', 'word', 'business', 'issue', 'side', 'kind',
    'head', 'house', 'service', 'friend', 'father', 'power', 'hour'
})

SIMPLE_WORD_MAP = {
    'utilize': 'use', 'implement': 'do', 'facilitate': 'help',
    'demonstrate': 'show', 'subsequently': 'then', 'consequently': 'so',
    'nevertheless': 'but', 'furthermore': 'also', 'approximately': 'about',
    'sufficient': 'enough', 'necessitate': 'need', 'initiate': 'start',
    'terminate': 'end', 'obtain': 'get', 'require': 'need', 'possess': 'have',
    'commence': 'begin', 'endeavor': 'try', 'ascertain': 'find out',
    'elucidate': 'explain', 'exemplify': 'show', 'methodology': 'method',
    'conceptualization': 'idea', 'phenomena': 'events', 'simultaneously': 'at the same time'
}

# ============================================================================
# SIMPLE TEXT PROCESSING (No NLTK dependency)
# ============================================================================

def tokenize_sentences(text: str) -> List[str]:
    """Simple sentence tokenization."""
    text = re.sub(r'\s+', ' ', text).strip()
    # Split on common sentence endings
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def tokenize_words(text: str) -> List[str]:
    """Simple word tokenization."""
    text = re.sub(r'\s+', ' ', text).strip().lower()
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return words

def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove common stopwords."""
    return [t for t in tokens if t.lower() not in STOPWORDS]

def simple_lemmatize(word: str) -> str:
    """Very basic lemmatization."""
    if word.endswith('ing') and len(word) > 5:
        return word[:-3]
    if word.endswith('ed') and len(word) > 4:
        return word[:-2]
    if word.endswith('s') and len(word) > 3 and not word.endswith('ss'):
        return word[:-1]
    return word

def count_syllables(word: str) -> int:
    """Count syllables in a word."""
    word = word.lower()
    vowels = 'aeiouy'
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith('e') and count > 1:
        count -= 1
    return max(1, count)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def calculate_avg_sentence_length(sentences: List[str]) -> Tuple[float, float]:
    if not sentences:
        return 0.0, 0.0
    lengths = [len(tokenize_words(sent)) for sent in sentences]
    avg_length = sum(lengths) / len(lengths)
    normalized = min(100, max(0, (avg_length - 10) * 5))
    return round(avg_length, 2), round(normalized, 2)

def calculate_vocabulary_complexity(tokens: List[str]) -> Tuple[float, float]:
    if not tokens:
        return 0.0, 0.0
    rare_count = 0
    for token in tokens:
        if token.lower() not in COMMON_WORDS:
            rare_count += 1
        for pattern in TECHNICAL_PATTERNS:
            if re.match(pattern, token, re.IGNORECASE):
                rare_count += 0.5
                break
    ratio = rare_count / len(tokens)
    normalized = min(100, ratio * 200)
    return round(ratio * 100, 2), round(normalized, 2)

def calculate_abstract_density(tokens: List[str]) -> Tuple[float, float]:
    if not tokens:
        return 0.0, 0.0
    abstract_count = 0
    for token in tokens:
        for suffix in ABSTRACT_SUFFIXES:
            if token.lower().endswith(suffix) and len(token) > len(suffix) + 2:
                abstract_count += 1
                break
    density = abstract_count / len(tokens) if tokens else 0
    normalized = min(100, density * 500)
    return round(density * 100, 2), round(normalized, 2)

def calculate_concept_repetition(lemmas: List[str]) -> Tuple[float, float]:
    if not lemmas:
        return 0.0, 0.0
    word_freq = Counter(lemmas)
    repeated = sum(1 for count in word_freq.values() if count > 1)
    total_unique = len(word_freq)
    if total_unique == 0:
        return 0.0, 0.0
    repetition_ratio = repeated / total_unique
    normalized = min(100, repetition_ratio * 200)
    return round(repetition_ratio * 100, 2), round(normalized, 2)

def calculate_readability(sentences: List[str], tokens: List[str]) -> Tuple[float, float]:
    if not sentences or not tokens:
        return 0.0, 50.0
    avg_sentence_length = len(tokens) / len(sentences)
    total_syllables = sum(count_syllables(word) for word in tokens)
    avg_syllables = total_syllables / len(tokens) if tokens else 0
    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
    flesch_score = max(0, min(100, flesch_score))
    cognitive_load = 100 - flesch_score
    return round(flesch_score, 2), round(cognitive_load, 2)

def get_feature_explanation(feature: str, score: float) -> str:
    """Generate student-friendly explanation."""
    explanations = {
        'sentence_length': {
            'low': "✅ Sentences are short and easy to follow",
            'medium': "📝 Sentences are medium length - stay focused",
            'high': "⚠️ Very long sentences - take your time reading"
        },
        'vocabulary_complexity': {
            'low': "✅ Uses everyday words you probably know",
            'medium': "📝 Some tricky words - look them up if needed",
            'high': "⚠️ Lots of hard words - keep a dictionary handy"
        },
        'abstract_density': {
            'low': "✅ Talks about concrete, easy-to-picture things",
            'medium': "📝 Mix of concrete and abstract ideas",
            'high': "⚠️ Many abstract ideas - try to find real examples"
        },
        'concept_repetition': {
            'low': "📝 Ideas aren't repeated much - take good notes",
            'medium': "✅ Good amount of repetition to help you learn",
            'high': "✅ Key ideas are repeated often - great for memory!"
        },
        'readability': {
            'low': "✅ Easy to read - like a conversation",
            'medium': "📝 Moderately challenging - normal textbook level",
            'high': "⚠️ Dense and complex - read slowly and carefully"
        }
    }
    level = 'low' if score < 35 else ('medium' if score < 65 else 'high')
    return explanations.get(feature, {}).get(level, "")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_text(text: str) -> Dict:
    """Main analysis function."""
    # Preprocess
    sentences = tokenize_sentences(text)
    tokens = tokenize_words(text)
    tokens_no_stop = remove_stopwords(tokens)
    lemmas = [simple_lemmatize(t) for t in tokens_no_stop]
    
    # Extract features
    sent_len_raw, sent_len_norm = calculate_avg_sentence_length(sentences)
    vocab_raw, vocab_norm = calculate_vocabulary_complexity(tokens)
    abstract_raw, abstract_norm = calculate_abstract_density(tokens)
    rep_raw, rep_norm = calculate_concept_repetition(lemmas)
    read_raw, read_norm = calculate_readability(sentences, tokens)
    
    features = {
        'sentence_length': {
            'raw': sent_len_raw, 'normalized': sent_len_norm,
            'label': 'Sentence Length', 'explanation': get_feature_explanation('sentence_length', sent_len_norm)
        },
        'vocabulary_complexity': {
            'raw': vocab_raw, 'normalized': vocab_norm,
            'label': 'Hard Words', 'explanation': get_feature_explanation('vocabulary_complexity', vocab_norm)
        },
        'abstract_density': {
            'raw': abstract_raw, 'normalized': abstract_norm,
            'label': 'Abstract Ideas', 'explanation': get_feature_explanation('abstract_density', abstract_norm)
        },
        'concept_repetition': {
            'raw': rep_raw, 'normalized': rep_norm,
            'label': 'Repetition', 'explanation': get_feature_explanation('concept_repetition', rep_norm)
        },
        'readability': {
            'raw': read_raw, 'normalized': read_norm,
            'label': 'Reading Difficulty', 'explanation': get_feature_explanation('readability', read_norm)
        }
    }
    
    # Calculate score
    score = 0.0
    for feature_name, weight in FEATURE_WEIGHTS.items():
        normalized_value = features[feature_name]['normalized']
        if weight < 0:
            score += abs(weight) * (100 - normalized_value)
        else:
            score += weight * normalized_value
    score = round(max(0, min(100, score)), 1)
    
    # Classify
    if score <= LOAD_THRESHOLDS['low']:
        classification = "Low"
    elif score <= LOAD_THRESHOLDS['medium']:
        classification = "Medium"
    else:
        classification = "High"
    
    # Interpretation
    interpretations = {
        "Low": {
            "emoji": "🎉", "title": "Easy to Learn!",
            "summary": f"Good news! This text scored {score}/100 on difficulty.",
            "meaning": "The text uses simple words and short sentences."
        },
        "Medium": {
            "emoji": "📚", "title": "Medium Difficulty",
            "summary": f"This text scored {score}/100 on difficulty.",
            "meaning": "Some complex parts. Pay attention and maybe re-read sections."
        },
        "High": {
            "emoji": "🧠", "title": "Challenging Content",
            "summary": f"Heads up! This text scored {score}/100 on difficulty.",
            "meaning": "Dense material with hard words. Take it slow."
        }
    }
    
    tips = {
        "Low": ["📖 You can read longer sections at once", "⏱️ 45-60 min study sessions work well"],
        "Medium": ["📖 Read in 25-30 min chunks", "✍️ Take notes on key points", "🔄 Review before moving on"],
        "High": ["📖 Read in 15-20 min chunks", "☕ Take 10-min breaks", "✍️ Summarize each paragraph", "🎥 Find videos on the topic"]
    }
    
    # Study plan
    word_count = len(tokens)
    reading_minutes = word_count / 200
    processing_mult = {'Low': 1.2, 'Medium': 1.5, 'High': 2.0}.get(classification, 1.5)
    total_study_time = reading_minutes * processing_mult
    session_lengths = {'Low': (45, 60), 'Medium': (25, 30), 'High': (15, 20)}
    min_session, max_session = session_lengths.get(classification, (25, 30))
    avg_session = (min_session + max_session) / 2
    num_sessions = max(1, round(total_study_time / avg_session))
    break_mins = 10 if classification == 'High' else 5
    total_with_breaks = total_study_time + (num_sessions - 1) * break_mins
    
    study_plan = {
        'total_words': word_count,
        'reading_time_mins': round(reading_minutes, 1),
        'study_time_mins': round(total_study_time, 1),
        'session_length': f"{min_session}-{max_session}",
        'num_sessions': num_sessions,
        'words_per_session': round(word_count / num_sessions),
        'break_mins': break_mins,
        'total_time_mins': round(total_with_breaks, 1)
    }
    
    # Simplify text
    simplified = text
    changes = []
    for complex_word, simple_word in SIMPLE_WORD_MAP.items():
        pattern = re.compile(re.escape(complex_word), re.IGNORECASE)
        if pattern.search(simplified):
            simplified = pattern.sub(simple_word, simplified)
            changes.append({'original': complex_word, 'simplified': simple_word})
    
    # Section analysis
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
    if not paragraphs:
        paragraphs = [text]
    
    sections = []
    for i, para in enumerate(paragraphs[:10]):  # Limit to 10 sections
        para_sentences = tokenize_sentences(para)
        para_tokens = tokenize_words(para)
        para_tokens_no_stop = remove_stopwords(para_tokens)
        para_lemmas = [simple_lemmatize(t) for t in para_tokens_no_stop]
        
        _, s_norm = calculate_avg_sentence_length(para_sentences)
        _, v_norm = calculate_vocabulary_complexity(para_tokens)
        _, a_norm = calculate_abstract_density(para_tokens)
        _, r_norm = calculate_concept_repetition(para_lemmas)
        _, rd_norm = calculate_readability(para_sentences, para_tokens)
        
        para_score = 0.0
        for fn, w in FEATURE_WEIGHTS.items():
            nv = {'sentence_length': s_norm, 'vocabulary_complexity': v_norm, 
                  'abstract_density': a_norm, 'concept_repetition': r_norm, 'readability': rd_norm}.get(fn, 50)
            para_score += abs(w) * (100 - nv) if w < 0 else w * nv
        para_score = round(max(0, min(100, para_score)), 1)
        
        sections.append({
            'index': i + 1,
            'preview': para[:150] + '...' if len(para) > 150 else para,
            'score': para_score,
            'word_count': len(para_tokens),
            'is_high_risk': para_score > 60
        })
    
    return {
        'score': score,
        'classification': classification,
        'interpretation': interpretations.get(classification, interpretations['Medium']),
        'tips': tips.get(classification, tips['Medium']),
        'features': features,
        'study_plan': study_plan,
        'simplified': {
            'text': simplified,
            'changes': changes,
            'changes_count': len(changes)
        },
        'sections': sections,
        'stats': {
            'word_count': word_count,
            'sentence_count': len(sentences),
            'unique_words': len(set(tokens))
        }
    }

# ============================================================================
# VERCEL HANDLER
# ============================================================================

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            text = data.get('text', '')
            
            if not text or len(text.split()) < 10:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Please provide at least 10 words'}).encode())
                return
            
            result = analyze_text(text)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
