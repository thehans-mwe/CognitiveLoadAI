"""
CognitiveLoad AI - Vercel Serverless API v3.0
==============================================
Analyzes educational text for cognitive load prediction.
Day 6: Refined analysis, caching, edge cases, performance.
"""

from http.server import BaseHTTPRequestHandler
import json
import re
import math
import hashlib
import time
from collections import Counter, OrderedDict
from typing import Dict, List, Tuple, Optional

# ============================================================================
# LRU CACHE (In-Memory, for Vercel cold/warm starts)
# ============================================================================

class LRUCache:
    """Simple LRU cache for analysis results."""
    def __init__(self, capacity: int = 64):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Dict):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

_analysis_cache = LRUCache(capacity=64)

def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# ============================================================================
# CONSTANTS
# ============================================================================

ABSTRACT_SUFFIXES = (
    'tion', 'sion', 'ness', 'ment', 'ity', 'ence', 'ance',
    'ship', 'dom', 'hood', 'ism', 'ology', 'phy', 'ics',
    'itude', 'iety', 'acy', 'esis', 'osis'
)

# Precompiled regex patterns for technical words
_TECHNICAL_PATTERNS_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\b\w+ization\b', r'\b\w+ification\b', r'\b\w+ological\b',
        r'\b\w+ometric\b', r'\b\w+aneous\b', r'\b\w+itious\b',
        r'\b\w+omorphic\b', r'\b\w+odynamic\b', r'\b\w+ogenesis\b',
        r'\b\w+ectomy\b', r'\b\w+oscopy\b', r'\b\w+ography\b'
    ]
]

# Precompiled utility regex
_RE_WHITESPACE = re.compile(r'\s+')
_RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')
_RE_WORDS = re.compile(r'\b[a-zA-Z]+\b')
_RE_ABBREVIATIONS = re.compile(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|e\.g|i\.e|St|Ave|Blvd)\.\s+', re.IGNORECASE)
_RE_ALL_CAPS_WORD = re.compile(r'^[A-Z]{2,}$')
_RE_CODE_BLOCK = re.compile(r'(```[\s\S]*?```|`[^`]+`|def\s+\w+|class\s+\w+|import\s+\w+|function\s+\w+|const\s+\w+|var\s+\w+|let\s+\w+)')
_RE_URL = re.compile(r'https?://\S+|www\.\S+')
_RE_NUMBER_HEAVY = re.compile(r'\d')
_RE_BULLET_LIST = re.compile(r'^\s*[-•*]\s+', re.MULTILINE)
_RE_NUMBERED_LIST = re.compile(r'^\s*\d+[.)]\s+', re.MULTILINE)
_RE_NULL_BYTES = re.compile(r'\x00')

FEATURE_WEIGHTS = {
    'sentence_length': 0.20,
    'vocabulary_complexity': 0.25,
    'abstract_density': 0.20,
    'concept_repetition': -0.10,
    'readability': 0.25
}

LOAD_THRESHOLDS = {'low': 35, 'medium': 65, 'high': 100}

# Common English stopwords
STOPWORDS = frozenset([
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

COMMON_WORDS = STOPWORDS | frozenset([
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
    'head', 'house', 'service', 'friend', 'father', 'power', 'hour',
    'also', 'like', 'back', 'would', 'could', 'still', 'much', 'well',
    'need', 'help', 'turn', 'start', 'show', 'play', 'run', 'move',
    'live', 'believe', 'hold', 'bring', 'happen', 'write', 'provide',
    'sit', 'stand', 'lose', 'pay', 'meet', 'include', 'continue',
    'set', 'learn', 'change', 'lead', 'understand', 'watch', 'follow',
    'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow',
    'open', 'walk', 'win', 'offer', 'remember', 'love', 'consider',
    'appear', 'buy', 'serve', 'die', 'send', 'expect', 'build', 'stay',
    'fall', 'cut', 'reach', 'kill', 'remain', 'body', 'keep', 'look',
    'form', 'begin', 'end', 'name', 'city', 'car', 'door', 'line',
    'face', 'girl', 'boy', 'age', 'food', 'game', 'heart', 'team'
])

# Expanded word simplification map (60+ entries)
SIMPLE_WORD_MAP = {
    'utilize': 'use', 'implement': 'carry out', 'facilitate': 'help',
    'demonstrate': 'show', 'subsequently': 'then', 'consequently': 'so',
    'nevertheless': 'but', 'furthermore': 'also', 'approximately': 'about',
    'sufficient': 'enough', 'necessitate': 'need', 'initiate': 'start',
    'terminate': 'end', 'obtain': 'get', 'require': 'need', 'possess': 'have',
    'commence': 'begin', 'endeavor': 'try', 'ascertain': 'find out',
    'elucidate': 'explain', 'exemplify': 'show', 'methodology': 'method',
    'conceptualization': 'idea', 'phenomena': 'events',
    'simultaneously': 'at the same time',
    'aforementioned': 'mentioned earlier', 'henceforth': 'from now on',
    'notwithstanding': 'despite', 'predominantly': 'mostly',
    'constitute': 'make up', 'encompass': 'include', 'augment': 'increase',
    'diminish': 'reduce', 'proliferate': 'spread', 'perpetuate': 'continue',
    'substantiate': 'prove', 'exacerbate': 'worsen', 'ameliorate': 'improve',
    'corroborate': 'confirm', 'disseminate': 'spread', 'delineate': 'describe',
    'juxtapose': 'compare', 'synthesize': 'combine', 'postulate': 'suggest',
    'extrapolate': 'extend', 'interpolate': 'estimate', 'predicate': 'base on',
    'promulgate': 'announce', 'adjudicate': 'judge', 'expound': 'explain',
    'ramification': 'result', 'paradigm': 'model', 'dichotomy': 'split',
    'ubiquitous': 'everywhere', 'heterogeneous': 'mixed', 'homogeneous': 'uniform',
    'superfluous': 'unnecessary', 'rudimentary': 'basic', 'quintessential': 'typical',
    'antithetical': 'opposite', 'pragmatic': 'practical', 'empirical': 'observed',
    'epistemological': 'knowledge-related', 'ontological': 'existence-related',
    'pedagogical': 'teaching-related', 'heuristic': 'rule of thumb',
    'efficacy': 'effectiveness', 'proclivity': 'tendency', 'propensity': 'tendency',
    'pertinent': 'relevant', 'salient': 'important', 'cogent': 'convincing',
    'succinct': 'brief', 'verbose': 'wordy', 'redundant': 'unnecessary',
    'ambiguous': 'unclear', 'concomitant': 'accompanying',
    'reconceptualization': 'rethinking', 'interconnectedness': 'connection',
    'competencies': 'skills', 'biosynthesis': 'creation in living things',
    'electrochemical': 'electric-chemical', 'phosphorylation': 'energy transfer process',
    'interdisciplinary': 'cross-field', 'disparate': 'different',
}

# Precompile simplification patterns for performance
_SIMPLE_WORD_PATTERNS = {
    word: (re.compile(re.escape(word), re.IGNORECASE), replacement)
    for word, replacement in SIMPLE_WORD_MAP.items()
}

# ============================================================================
# TEXT PROCESSING (Optimized, No NLTK)
# ============================================================================

def detect_text_type(text: str) -> Dict[str, bool]:
    """Detect special characteristics of the input text."""
    words = text.split()
    total_chars = len(text)
    if total_chars == 0:
        return {'is_code': False, 'is_all_caps': False, 'is_number_heavy': False,
                'has_lists': False, 'has_urls': False, 'is_repetitive': False}

    # Code detection
    code_indicators = len(_RE_CODE_BLOCK.findall(text))
    is_code = code_indicators >= 3

    # All-caps detection
    caps_words = sum(1 for w in words if _RE_ALL_CAPS_WORD.match(w) and len(w) > 1)
    is_all_caps = len(words) > 0 and caps_words / len(words) > 0.5

    # Number-heavy detection
    digit_chars = len(_RE_NUMBER_HEAVY.findall(text))
    is_number_heavy = digit_chars / total_chars > 0.3 if total_chars else False

    # List detection
    has_lists = bool(_RE_BULLET_LIST.search(text)) or bool(_RE_NUMBERED_LIST.search(text))

    # URL detection
    has_urls = bool(_RE_URL.search(text))

    # Repetition detection (duplicate sentences)
    sentences = tokenize_sentences(text)
    unique_sents = set(s.lower().strip() for s in sentences)
    is_repetitive = len(sentences) > 3 and len(unique_sents) / len(sentences) < 0.5

    return {
        'is_code': is_code, 'is_all_caps': is_all_caps,
        'is_number_heavy': is_number_heavy, 'has_lists': has_lists,
        'has_urls': has_urls, 'is_repetitive': is_repetitive
    }


def preprocess_text(text: str) -> str:
    """Clean text for analysis, handling edge cases."""
    # Remove URLs
    text = _RE_URL.sub('', text)
    # Handle all-caps by converting to sentence case
    words = text.split()
    if words:
        caps_words = sum(1 for w in words if _RE_ALL_CAPS_WORD.match(w) and len(w) > 1)
        if len(words) > 0 and caps_words / len(words) > 0.5:
            # Convert to title case for better analysis
            sentences = text.split('. ')
            text = '. '.join(s.capitalize() if s else s for s in sentences)
    # Protect abbreviations from bad sentence splitting
    text = _RE_ABBREVIATIONS.sub(lambda m: m.group().replace('. ', '.\u200B '), text)
    return text


def tokenize_sentences(text: str) -> List[str]:
    """Improved sentence tokenization with abbreviation handling."""
    text = _RE_WHITESPACE.sub(' ', text).strip()
    if not text:
        return []
    sentences = _RE_SENTENCE_SPLIT.split(text)
    result = []
    for s in sentences:
        s = s.strip().replace('\u200B', '')
        if s and len(s) > 2:
            result.append(s)
    # If no splits found, treat entire text as one sentence
    if not result and text:
        result = [text]
    return result


def tokenize_words(text: str) -> List[str]:
    """Optimized word tokenization."""
    return _RE_WORDS.findall(text.lower())


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS]


def simple_lemmatize(word: str) -> str:
    """Basic lemmatization with improved suffix handling."""
    w = word.lower()
    if w.endswith('ies') and len(w) > 4:
        return w[:-3] + 'y'
    if w.endswith('ing') and len(w) > 5:
        base = w[:-3]
        if base and base[-1] == base[-2:][0:1]:  # running -> run
            return base[:-1]
        return base
    if w.endswith('ed') and len(w) > 4:
        return w[:-2]
    if w.endswith('ly') and len(w) > 4:
        return w[:-2]
    if w.endswith('s') and len(w) > 3 and not w.endswith('ss') and not w.endswith('us'):
        return w[:-1]
    return w


def count_syllables(word: str) -> int:
    """Count syllables using vowel-group method."""
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
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
    return max(1, count)

# ============================================================================
# FEATURE EXTRACTION (Optimized)
# ============================================================================

def calculate_avg_sentence_length(sentences: List[str]) -> Tuple[float, float]:
    if not sentences:
        return 0.0, 0.0
    lengths = [len(tokenize_words(sent)) for sent in sentences]
    avg_length = sum(lengths) / len(lengths)
    # Improved normalization curve (sigmoid-like)
    normalized = min(100, max(0, (avg_length - 8) * 4.5))
    return round(avg_length, 2), round(normalized, 2)


def calculate_vocabulary_complexity(tokens: List[str]) -> Tuple[float, float]:
    if not tokens:
        return 0.0, 0.0
    rare_count = 0
    token_set = set(t.lower() for t in tokens)
    # Batch: check unique tokens only  
    uncommon_tokens = token_set - COMMON_WORDS
    rare_count = sum(1 for t in tokens if t.lower() in uncommon_tokens)
    # Technical pattern bonus (check unique tokens only)
    tech_bonus = 0
    for token in uncommon_tokens:
        for pat in _TECHNICAL_PATTERNS_COMPILED:
            if pat.match(token):
                tech_bonus += tokens.count(token) * 0.5
                break
    rare_count += tech_bonus
    ratio = rare_count / len(tokens)
    normalized = min(100, ratio * 180)
    return round(ratio * 100, 2), round(normalized, 2)


def calculate_abstract_density(tokens: List[str]) -> Tuple[float, float]:
    if not tokens:
        return 0.0, 0.0
    abstract_count = 0
    for token in tokens:
        lower = token.lower()
        for suffix in ABSTRACT_SUFFIXES:
            if lower.endswith(suffix) and len(lower) > len(suffix) + 2:
                abstract_count += 1
                break
    density = abstract_count / len(tokens)
    normalized = min(100, density * 450)
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
    total_syllables = sum(count_syllables(w) for w in tokens)
    avg_syllables = total_syllables / len(tokens) if tokens else 0
    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
    flesch_score = max(0, min(100, flesch_score))
    cognitive_load = 100 - flesch_score
    return round(flesch_score, 2), round(cognitive_load, 2)


def get_feature_explanation(feature: str, score: float, raw: float) -> str:
    """Generate contextual, data-driven explanation."""
    level = 'low' if score < 35 else ('medium' if score < 65 else 'high')

    explanations = {
        'sentence_length': {
            'low': f"✅ Avg {raw} words/sentence — short and easy to follow",
            'medium': f"📝 Avg {raw} words/sentence — moderate length, stay focused",
            'high': f"⚠️ Avg {raw} words/sentence — very long, break into chunks mentally"
        },
        'vocabulary_complexity': {
            'low': f"✅ {raw:.0f}% uncommon words — everyday vocabulary",
            'medium': f"📝 {raw:.0f}% uncommon words — some technical terms to learn",
            'high': f"⚠️ {raw:.0f}% uncommon words — heavy jargon, keep a glossary"
        },
        'abstract_density': {
            'low': f"✅ {raw:.1f}% abstract terms — concrete and visual",
            'medium': f"📝 {raw:.1f}% abstract terms — mix of concrete and abstract ideas",
            'high': f"⚠️ {raw:.1f}% abstract terms — many concepts to visualize"
        },
        'concept_repetition': {
            'low': f"📝 {raw:.0f}% terms repeat — few reinforcements, take notes",
            'medium': f"✅ {raw:.0f}% terms repeat — good reinforcement for learning",
            'high': f"✅ {raw:.0f}% terms repeat — strong repetition, great for memory"
        },
        'readability': {
            'low': f"✅ Flesch score {raw} — reads like a conversation",
            'medium': f"📝 Flesch score {raw} — standard textbook readability",
            'high': f"⚠️ Flesch score {raw} — dense academic prose, read slowly"
        }
    }
    return explanations.get(feature, {}).get(level, "")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_text(text: str) -> Dict:
    """Main analysis function with caching and edge-case handling."""
    t_start = time.time()

    # Check cache
    text_key = _text_hash(text)
    cached = _analysis_cache.get(text_key)
    if cached:
        cached['_meta'] = {
            'cached': True,
            'processing_ms': round((time.time() - t_start) * 1000, 1),
            'cache_hits': _analysis_cache.hits,
            'cache_misses': _analysis_cache.misses
        }
        return cached

    # Detect text characteristics for edge-case adjustments
    text_type = detect_text_type(text)

    # Preprocess
    cleaned = preprocess_text(text)
    sentences = tokenize_sentences(cleaned)
    tokens = tokenize_words(cleaned)
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
            'label': 'Sentence Length',
            'explanation': get_feature_explanation('sentence_length', sent_len_norm, sent_len_raw)
        },
        'vocabulary_complexity': {
            'raw': vocab_raw, 'normalized': vocab_norm,
            'label': 'Hard Words',
            'explanation': get_feature_explanation('vocabulary_complexity', vocab_norm, vocab_raw)
        },
        'abstract_density': {
            'raw': abstract_raw, 'normalized': abstract_norm,
            'label': 'Abstract Ideas',
            'explanation': get_feature_explanation('abstract_density', abstract_norm, abstract_raw)
        },
        'concept_repetition': {
            'raw': rep_raw, 'normalized': rep_norm,
            'label': 'Repetition',
            'explanation': get_feature_explanation('concept_repetition', rep_norm, rep_raw)
        },
        'readability': {
            'raw': read_raw, 'normalized': read_norm,
            'label': 'Reading Difficulty',
            'explanation': get_feature_explanation('readability', read_norm, read_raw)
        }
    }

    # Calculate weighted score
    score = 0.0
    for feature_name, weight in FEATURE_WEIGHTS.items():
        nv = features[feature_name]['normalized']
        if weight < 0:
            score += abs(weight) * (100 - nv)
        else:
            score += weight * nv
    score = round(max(0, min(100, score)), 1)

    # Edge-case adjustments: warn about special text types
    warnings = []
    if text_type['is_code']:
        warnings.append("This text contains code — scores may not reflect reading difficulty accurately.")
    if text_type['is_number_heavy']:
        warnings.append("Number-heavy text detected — cognitive load from data interpretation not fully captured.")
    if text_type['is_repetitive']:
        warnings.append("Highly repetitive content detected — actual difficulty may be lower than scored.")
    if text_type['is_all_caps']:
        warnings.append("All-caps text was normalized for analysis — readability may differ from visual presentation.")

    # Classify
    if score <= LOAD_THRESHOLDS['low']:
        classification = "Low"
    elif score <= LOAD_THRESHOLDS['medium']:
        classification = "Medium"
    else:
        classification = "High"

    # Find the dominant difficulty factor
    feature_scores = {k: v['normalized'] for k, v in features.items() if FEATURE_WEIGHTS.get(k, 0) > 0}
    dominant_feature = max(feature_scores, key=feature_scores.get) if feature_scores else 'readability'
    dominant_label = features[dominant_feature]['label']

    # Richer, contextual interpretations
    word_count = len(tokens)
    interpretations = {
        "Low": {
            "emoji": "🎉", "title": "Easy to Learn!",
            "summary": f"Great news! This text scored {score}/100 on difficulty.",
            "meaning": (
                f"With {word_count} words across {len(sentences)} sentence{'s' if len(sentences) != 1 else ''}, "
                f"this material uses straightforward language and clear structure. "
                f"You should be able to read and understand it at a normal pace. "
                f"The main factor was '{dominant_label}' — but overall, nothing too demanding."
            )
        },
        "Medium": {
            "emoji": "📚", "title": "Medium Difficulty",
            "summary": f"This text scored {score}/100 on difficulty.",
            "meaning": (
                f"At {word_count} words and {len(sentences)} sentence{'s' if len(sentences) != 1 else ''}, "
                f"this material has some challenging elements. "
                f"'{dominant_label}' is the biggest contributor to difficulty. "
                f"Take notes and re-read tricky sections to make sure you absorb it."
            )
        },
        "High": {
            "emoji": "🧠", "title": "Challenging Content",
            "summary": f"Heads up! This text scored {score}/100 on difficulty.",
            "meaning": (
                f"This is dense material — {word_count} words across {len(sentences)} sentence{'s' if len(sentences) != 1 else ''}. "
                f"'{dominant_label}' is the primary difficulty driver. "
                f"Expect to read sections multiple times. "
                f"Break it into small chunks, take frequent breaks, and use external resources."
            )
        }
    }

    # Smarter tips based on which features scored highest
    tips = _generate_contextual_tips(classification, features, text_type)

    # Study plan (improved accuracy)
    reading_wpm = 200  # average reading speed
    reading_minutes = word_count / reading_wpm

    # Study time includes re-reading, note-taking, comprehension pauses
    # Higher cognitive load = more time per word
    processing_mult = {'Low': 1.5, 'Medium': 2.5, 'High': 4.0}.get(classification, 2.5)
    raw_study_time = reading_minutes * processing_mult

    # Enforce sensible minimums based on difficulty
    min_study_time = {'Low': 5, 'Medium': 10, 'High': 15}.get(classification, 10)
    total_study_time = max(min_study_time, raw_study_time)

    # Session lengths (minutes of focused study before a break)
    session_lengths = {'Low': (25, 35), 'Medium': (20, 25), 'High': (10, 15)}
    min_session, max_session = session_lengths.get(classification, (20, 25))
    avg_session = (min_session + max_session) / 2
    num_sessions = max(1, math.ceil(total_study_time / avg_session))

    # Break duration
    break_mins = 10 if classification == 'High' else (7 if classification == 'Medium' else 5)

    # Recalculate session length to evenly distribute time
    actual_session_length = round(total_study_time / num_sessions)
    total_with_breaks = total_study_time + max(0, num_sessions - 1) * break_mins

    study_plan = {
        'total_words': word_count,
        'reading_time_mins': round(reading_minutes, 1),
        'study_time_mins': round(total_study_time, 1),
        'session_length': f"{actual_session_length}",
        'num_sessions': num_sessions,
        'words_per_session': round(word_count / num_sessions),
        'break_mins': break_mins,
        'total_time_mins': round(total_with_breaks, 1)
    }

    # Simplify text (using precompiled patterns)
    simplified = text
    changes = []
    for word, (pattern, replacement) in _SIMPLE_WORD_PATTERNS.items():
        if pattern.search(simplified):
            simplified = pattern.sub(replacement, simplified)
            changes.append({'original': word, 'simplified': replacement})

    # Section analysis — improved paragraph splitting
    paragraphs = _split_into_sections(text)
    sections = []
    for i, para in enumerate(paragraphs[:12]):  # Limit to 12 sections
        para_sentences = tokenize_sentences(para)
        para_tokens = tokenize_words(para)
        if len(para_tokens) < 3:
            continue
        para_tokens_no_stop = remove_stopwords(para_tokens)
        para_lemmas = [simple_lemmatize(t) for t in para_tokens_no_stop]

        _, s_norm = calculate_avg_sentence_length(para_sentences)
        _, v_norm = calculate_vocabulary_complexity(para_tokens)
        _, a_norm = calculate_abstract_density(para_tokens)
        _, r_norm = calculate_concept_repetition(para_lemmas)
        _, rd_norm = calculate_readability(para_sentences, para_tokens)

        norms = {
            'sentence_length': s_norm, 'vocabulary_complexity': v_norm,
            'abstract_density': a_norm, 'concept_repetition': r_norm,
            'readability': rd_norm
        }
        para_score = 0.0
        for fn, w in FEATURE_WEIGHTS.items():
            nv = norms.get(fn, 50)
            para_score += abs(w) * (100 - nv) if w < 0 else w * nv
        para_score = round(max(0, min(100, para_score)), 1)

        sections.append({
            'index': i + 1,
            'preview': para[:150] + '...' if len(para) > 150 else para,
            'score': para_score,
            'word_count': len(para_tokens),
            'is_high_risk': para_score > 60
        })

    t_end = time.time()
    processing_ms = round((t_end - t_start) * 1000, 1)

    result = {
        'score': score,
        'classification': classification,
        'interpretation': interpretations.get(classification, interpretations['Medium']),
        'tips': tips,
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
            'unique_words': len(set(tokens)),
            'avg_word_length': round(sum(len(t) for t in tokens) / max(1, len(tokens)), 1)
        },
        'warnings': warnings,
        'text_type': text_type,
        '_meta': {
            'cached': False,
            'processing_ms': processing_ms,
            'version': '3.0.0'
        }
    }

    # Cache the result
    _analysis_cache.put(text_key, result)
    return result


def _generate_contextual_tips(classification: str, features: Dict, text_type: Dict) -> List[str]:
    """Generate tips based on actual feature analysis, not just classification."""
    tips = []
    # Always include a general study strategy tip
    strategy = {
        "Low": "📖 Read at your normal pace — this is approachable material",
        "Medium": "📖 Active reading recommended — pause and summarize every few paragraphs",
        "High": "📖 Use SQ3R method: Survey, Question, Read, Recite, Review"
    }
    tips.append(strategy.get(classification, strategy["Medium"]))

    # Feature-specific tips (only for the problems that matter)
    if features['sentence_length']['normalized'] > 55:
        tips.append("✂️ Long sentences detected — try mentally breaking them at commas and conjunctions")
    if features['vocabulary_complexity']['normalized'] > 55:
        tips.append("📚 Many technical terms — create a vocabulary list before deep reading")
    if features['abstract_density']['normalized'] > 55:
        tips.append("🎨 High abstraction — draw diagrams or find concrete examples for abstract concepts")
    if features['concept_repetition']['normalized'] < 30:
        tips.append("📝 Low repetition — key ideas appear only once, so take detailed notes")
    if features['readability']['normalized'] > 60:
        tips.append("🔍 Dense prose — read one paragraph at a time, then paraphrase it in your own words")

    # Edge-case tips
    if text_type.get('is_code'):
        tips.append("💻 Code detected — try running examples to understand logic, not just reading")
    if text_type.get('has_lists'):
        tips.append("📋 Lists found — use them as a natural checklist for study progress")
    if text_type.get('is_number_heavy'):
        tips.append("🔢 Number-heavy content — work through calculations step by step")

    # Session / break tips based on classification
    session_tips = {
        "Low": "⏱️ 45-60 min study sessions work well with 5 min breaks",
        "Medium": "⏱️ Try 25-30 min focused sessions with 7 min breaks (Pomodoro style)",
        "High": "⏱️ 15-20 min sessions max — your brain needs 10 min breaks for this material"
    }
    tips.append(session_tips.get(classification, session_tips["Medium"]))

    # Review tip
    if classification in ("Medium", "High"):
        tips.append("🔄 Review your notes within 24 hours — spaced repetition helps lock in tough material")

    return tips


def _split_into_sections(text: str) -> List[str]:
    """Smart section splitting: paragraphs first, then long-sentence fallback."""
    # Try double-newline paragraph splits
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
    if len(paragraphs) >= 2:
        return paragraphs

    # Try single-newline splits
    paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) > 30]
    if len(paragraphs) >= 2:
        return paragraphs

    # Fallback: split long text into ~100-word chunks by sentence boundaries
    sentences = tokenize_sentences(text)
    if len(sentences) <= 2:
        return [text]

    chunks = []
    current_chunk = []
    current_words = 0
    for sent in sentences:
        word_count = len(sent.split())
        if current_words + word_count > 100 and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sent]
            current_words = word_count
        else:
            current_chunk.append(sent)
            current_words += word_count
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks if len(chunks) >= 2 else [text]

# ============================================================================
# VERCEL HANDLER
# ============================================================================

MAX_TEXT_LENGTH = 50000
MIN_WORD_COUNT = 10


def sanitize_text(text: str) -> str:
    """Clean and sanitize input text."""
    if not isinstance(text, str):
        return ''
    text = text.strip()
    text = _RE_NULL_BYTES.sub('', text)
    # Collapse extreme whitespace but preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    return text


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: dict):
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
        """Health check with cache stats."""
        self._send_json(200, {
            'status': 'healthy',
            'service': 'CognitiveLoad AI',
            'version': '3.0.0',
            'cache': {
                'entries': len(_analysis_cache.cache),
                'hits': _analysis_cache.hits,
                'misses': _analysis_cache.misses
            },
            'endpoints': {'POST /api/analyze': 'Analyze text for cognitive load'}
        })

    def do_POST(self):
        t_start = time.time()
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_json(400, {'error': 'Empty request body', 'code': 'EMPTY_BODY'})
                return
            if content_length > MAX_TEXT_LENGTH * 2:
                self._send_json(413, {
                    'error': 'Request body too large. Max ~50,000 characters.',
                    'code': 'BODY_TOO_LARGE',
                    'suggestion': 'Try splitting your text into smaller sections.'
                })
                return

            body = self.rfile.read(content_length).decode('utf-8')

            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                self._send_json(400, {
                    'error': 'Invalid JSON in request body',
                    'code': 'INVALID_JSON',
                    'detail': str(e)
                })
                return

            if not isinstance(data, dict):
                self._send_json(400, {'error': 'Request body must be a JSON object', 'code': 'INVALID_FORMAT'})
                return

            text = data.get('text', '')
            text = sanitize_text(text)

            if not text:
                self._send_json(400, {
                    'error': 'Please provide text in the "text" field',
                    'code': 'MISSING_TEXT',
                    'suggestion': 'Send a JSON body like: {"text": "Your study material here..."}'
                })
                return

            word_count = len(text.split())
            if word_count < MIN_WORD_COUNT:
                self._send_json(400, {
                    'error': f'Text too short. Please provide at least {MIN_WORD_COUNT} words (got {word_count}).',
                    'code': 'TEXT_TOO_SHORT',
                    'word_count': word_count,
                    'min_words': MIN_WORD_COUNT,
                    'suggestion': 'Try pasting a full paragraph or more for meaningful analysis.'
                })
                return

            result = analyze_text(text)
            self._send_json(200, result)

        except Exception as e:
            elapsed = round((time.time() - t_start) * 1000, 1)
            self._send_json(500, {
                'error': 'Internal server error. Please try again.',
                'code': 'INTERNAL_ERROR',
                'processing_ms': elapsed,
                'suggestion': 'If this persists, try with simpler or shorter text.'
            })
