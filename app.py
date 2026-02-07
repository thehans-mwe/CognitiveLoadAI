"""
CognitiveLoad AI - Student Cognitive Overload Prediction System
================================================================
A Streamlit web application that analyzes educational text to predict
cognitive load and help students optimize their learning experience.

Author: CognitiveLoad AI Team
Version: 2.0.0
License: MIT

Design Philosophy:
- Ethical, transparent AI
- No personal data storage
- Educational focus - helping students learn better
- Simple, student-friendly explanations
"""

import streamlit as st
import nltk
import re
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# NLTK RESOURCE MANAGEMENT
# ============================================================================

@st.cache_resource
def download_nltk_resources():
    """Download all required NLTK resources for NLP processing."""
    resources = [
        'punkt', 
        'punkt_tab',
        'stopwords', 
        'wordnet', 
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng'
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            st.warning(f"Could not download {resource}: {e}")
    return True

# Initialize NLTK resources
download_nltk_resources()

# Import NLTK modules after download
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# ============================================================================
# CONSTANTS AND CONFIGURATION
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

# Light mode color scheme
COLORS = {
    'low': '#10b981',       # Emerald green
    'medium': '#f59e0b',    # Amber
    'high': '#ef4444',      # Red
    'primary': '#3b82f6',   # Blue
    'secondary': '#6b7280', # Gray
    'background': '#ffffff',
    'surface': '#f8fafc',
    'text': '#1e293b',
    'text_secondary': '#64748b'
}

# Simple word replacements for text simplification
SIMPLE_WORD_MAP = {
    'utilize': 'use',
    'implement': 'do',
    'facilitate': 'help',
    'demonstrate': 'show',
    'subsequently': 'then',
    'consequently': 'so',
    'nevertheless': 'but',
    'furthermore': 'also',
    'approximately': 'about',
    'sufficient': 'enough',
    'necessitate': 'need',
    'initiate': 'start',
    'terminate': 'end',
    'obtain': 'get',
    'require': 'need',
    'possess': 'have',
    'commence': 'begin',
    'endeavor': 'try',
    'ascertain': 'find out',
    'elucidate': 'explain',
    'exemplify': 'show',
    'methodology': 'method',
    'conceptualization': 'idea',
    'reconceptualization': 'new way of thinking',
    'epistemological': 'about knowledge',
    'ramifications': 'effects',
    'indeterminacy': 'uncertainty',
    'paradigms': 'ways of thinking',
    'instrumentalist': 'practical',
    'ontology': 'nature of reality',
    'formalism': 'formal rules',
    'apparatus': 'tool',
    'metaphysical': 'philosophical',
    'phenomena': 'events',
    'decoherence': 'loss of quantum effects',
    'substrates': 'base materials',
    'entanglement': 'connection',
    'interpretive': 'understanding',
    'assumptions': 'beliefs',
    'conjugate': 'paired',
    'simultaneously': 'at the same time',
    'intrinsic': 'built-in',
    'predictive': 'forecasting',
    'capabilities': 'abilities',
    'enzymatic': 'enzyme-based',
    'chlorophyll': 'green plant chemical',
    'photosynthesis': 'how plants make food from sunlight',
}

# ============================================================================
# NLP PREPROCESSING CLASS
# ============================================================================

class TextPreprocessor:
    """Handles all NLP preprocessing tasks for educational text analysis."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def tokenize_sentences(self, text: str) -> List[str]:
        text = self._clean_text(text)
        return sent_tokenize(text)
    
    def tokenize_words(self, text: str) -> List[str]:
        text = self._clean_text(text)
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token.isalpha()]
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def get_pos_tags(self, tokens: List[str]) -> List[Tuple[str, str]]:
        return pos_tag(tokens)
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def full_preprocess(self, text: str) -> Dict:
        sentences = self.tokenize_sentences(text)
        tokens = self.tokenize_words(text)
        tokens_no_stopwords = self.remove_stopwords(tokens)
        lemmas = self.lemmatize(tokens_no_stopwords)
        pos_tags = self.get_pos_tags(tokens)
        
        return {
            'sentences': sentences,
            'tokens': tokens,
            'tokens_no_stopwords': tokens_no_stopwords,
            'lemmas': lemmas,
            'pos_tags': pos_tags,
            'word_count': len(tokens),
            'sentence_count': len(sentences),
            'unique_words': len(set(tokens))
        }

# ============================================================================
# TEXT SIMPLIFIER CLASS
# ============================================================================

class TextSimplifier:
    """
    Simplifies complex text into easier-to-understand language.
    Uses word replacement and sentence restructuring.
    """
    
    def __init__(self):
        self.word_map = SIMPLE_WORD_MAP
        self.preprocessor = TextPreprocessor()
    
    def simplify_text(self, text: str) -> Dict:
        """
        Simplify the input text and return both simplified version
        and a list of changes made.
        """
        simplified = text
        changes = []
        
        # Replace complex words with simpler alternatives
        for complex_word, simple_word in self.word_map.items():
            pattern = re.compile(re.escape(complex_word), re.IGNORECASE)
            if pattern.search(simplified):
                simplified = pattern.sub(simple_word, simplified)
                changes.append({
                    'original': complex_word,
                    'simplified': simple_word,
                    'reason': f'"{complex_word}" is a complex word'
                })
        
        # Break up very long sentences
        sentences = sent_tokenize(simplified)
        new_sentences = []
        
        for sent in sentences:
            words = word_tokenize(sent)
            if len(words) > 25:
                # Try to split at conjunctions or commas
                parts = re.split(r'(?:,\s*(?:and|but|or|which|that|because|although|however)|\s+(?:and|but|or)\s+)', sent)
                if len(parts) > 1:
                    for part in parts:
                        part = part.strip()
                        if part and len(part) > 10:
                            if not part[0].isupper():
                                part = part[0].upper() + part[1:]
                            if not part.endswith('.'):
                                part += '.'
                            new_sentences.append(part)
                    changes.append({
                        'original': sent[:50] + '...',
                        'simplified': 'Split into shorter sentences',
                        'reason': 'Sentence was too long'
                    })
                else:
                    new_sentences.append(sent)
            else:
                new_sentences.append(sent)
        
        simplified = ' '.join(new_sentences)
        
        return {
            'original': text,
            'simplified': simplified,
            'changes': changes,
            'word_changes_count': len([c for c in changes if 'complex word' in c.get('reason', '')]),
            'sentence_changes_count': len([c for c in changes if 'too long' in c.get('reason', '')])
        }
    
    def get_simple_explanation(self, text: str, max_sentences: int = 3) -> str:
        """
        Generate a very simple summary/explanation of the text.
        """
        sentences = sent_tokenize(text)
        
        # Get the first few sentences as a base
        if len(sentences) <= max_sentences:
            base = ' '.join(sentences)
        else:
            base = ' '.join(sentences[:max_sentences])
        
        # Simplify this base
        result = self.simplify_text(base)
        return result['simplified']

# ============================================================================
# FEATURE EXTRACTION CLASS
# ============================================================================

class CognitiveFeatureExtractor:
    """Extracts cognitive load features from preprocessed text."""
    
    def __init__(self):
        self.common_words = self._load_common_words()
        
    def _load_common_words(self) -> set:
        common = set(stopwords.words('english'))
        basic_words = {
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
        }
        return common.union(basic_words)
    
    def calculate_avg_sentence_length(self, sentences: List[str]) -> Tuple[float, float]:
        if not sentences:
            return 0.0, 0.0
        lengths = [len(word_tokenize(sent)) for sent in sentences]
        avg_length = sum(lengths) / len(lengths)
        normalized = min(100, max(0, (avg_length - 10) * 5))
        return round(avg_length, 2), round(normalized, 2)
    
    def calculate_vocabulary_complexity(self, tokens: List[str]) -> Tuple[float, float]:
        if not tokens:
            return 0.0, 0.0
        rare_count = 0
        for token in tokens:
            if token.lower() not in self.common_words:
                rare_count += 1
            for pattern in TECHNICAL_PATTERNS:
                if re.match(pattern, token, re.IGNORECASE):
                    rare_count += 0.5
                    break
        ratio = rare_count / len(tokens)
        normalized = min(100, ratio * 200)
        return round(ratio * 100, 2), round(normalized, 2)
    
    def calculate_abstract_density(self, tokens: List[str], pos_tags: List[Tuple]) -> Tuple[float, float]:
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
    
    def calculate_concept_repetition(self, lemmas: List[str]) -> Tuple[float, float]:
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
    
    def calculate_readability(self, text: str, sentences: List[str], tokens: List[str]) -> Tuple[float, float]:
        if not sentences or not tokens:
            return 0.0, 50.0
        avg_sentence_length = len(tokens) / len(sentences)
        total_syllables = sum(self._count_syllables(word) for word in tokens)
        avg_syllables = total_syllables / len(tokens) if tokens else 0
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        flesch_score = max(0, min(100, flesch_score))
        cognitive_load = 100 - flesch_score
        return round(flesch_score, 2), round(cognitive_load, 2)
    
    def _count_syllables(self, word: str) -> int:
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
    
    def extract_all_features(self, preprocessed: Dict, original_text: str) -> Dict:
        sent_len_raw, sent_len_norm = self.calculate_avg_sentence_length(preprocessed['sentences'])
        vocab_raw, vocab_norm = self.calculate_vocabulary_complexity(preprocessed['tokens'])
        abstract_raw, abstract_norm = self.calculate_abstract_density(preprocessed['tokens'], preprocessed['pos_tags'])
        rep_raw, rep_norm = self.calculate_concept_repetition(preprocessed['lemmas'])
        read_raw, read_norm = self.calculate_readability(original_text, preprocessed['sentences'], preprocessed['tokens'])
        
        return {
            'sentence_length': {
                'raw': sent_len_raw,
                'normalized': sent_len_norm,
                'unit': 'words/sentence',
                'description': 'How long each sentence is on average',
                'simple_explanation': self._get_simple_feature_explanation('sentence_length', sent_len_norm)
            },
            'vocabulary_complexity': {
                'raw': vocab_raw,
                'normalized': vocab_norm,
                'unit': '% hard words',
                'description': 'How many difficult words are used',
                'simple_explanation': self._get_simple_feature_explanation('vocabulary_complexity', vocab_norm)
            },
            'abstract_density': {
                'raw': abstract_raw,
                'normalized': abstract_norm,
                'unit': '% abstract ideas',
                'description': 'How many abstract concepts (hard to picture)',
                'simple_explanation': self._get_simple_feature_explanation('abstract_density', abstract_norm)
            },
            'concept_repetition': {
                'raw': rep_raw,
                'normalized': rep_norm,
                'unit': '% repeated ideas',
                'description': 'How often key ideas are repeated (good for learning!)',
                'simple_explanation': self._get_simple_feature_explanation('concept_repetition', rep_norm)
            },
            'readability': {
                'raw': read_raw,
                'normalized': read_norm,
                'unit': 'ease score',
                'description': 'Overall how easy the text is to read',
                'simple_explanation': self._get_simple_feature_explanation('readability', read_norm)
            }
        }
    
    def _get_simple_feature_explanation(self, feature: str, score: float) -> str:
        """Generate a simple, student-friendly explanation for each feature."""
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
        
        if score < 35:
            level = 'low'
        elif score < 65:
            level = 'medium'
        else:
            level = 'high'
            
        return explanations.get(feature, {}).get(level, "")

# ============================================================================
# COGNITIVE LOAD CALCULATOR
# ============================================================================

class CognitiveLoadCalculator:
    """Calculates overall cognitive load score from extracted features."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or FEATURE_WEIGHTS
        
    def calculate_score(self, features: Dict) -> float:
        score = 0.0
        for feature_name, weight in self.weights.items():
            if feature_name in features:
                normalized_value = features[feature_name]['normalized']
                if weight < 0:
                    contribution = abs(weight) * (100 - normalized_value)
                else:
                    contribution = weight * normalized_value
                score += contribution
        return round(max(0, min(100, score)), 1)
    
    def classify_load(self, score: float) -> str:
        if score <= LOAD_THRESHOLDS['low']:
            return "Low"
        elif score <= LOAD_THRESHOLDS['medium']:
            return "Medium"
        else:
            return "High"
    
    def get_load_color(self, classification: str) -> str:
        return COLORS.get(classification.lower(), COLORS['secondary'])
    
    def get_simple_interpretation(self, score: float, classification: str, features: Dict) -> Dict:
        """
        Generate a simple, student-friendly interpretation.
        Returns a dictionary with different sections.
        """
        # Main message based on classification
        main_messages = {
            "Low": {
                "emoji": "🎉",
                "title": "Easy to Learn!",
                "summary": f"Good news! This text scored {score}/100 on difficulty. It's pretty straightforward.",
                "meaning": "This means the text uses simple words and short sentences. You should be able to understand it without too much trouble."
            },
            "Medium": {
                "emoji": "📚",
                "title": "Medium Difficulty",
                "summary": f"This text scored {score}/100 on difficulty. It needs some focus.",
                "meaning": "The text has some complex parts. You'll need to pay attention and maybe re-read some sections."
            },
            "High": {
                "emoji": "🧠",
                "title": "Challenging Content",
                "summary": f"Heads up! This text scored {score}/100 on difficulty. It's quite complex.",
                "meaning": "This is dense material with hard words and complex ideas. Don't worry - just take it slow and break it into small pieces."
            }
        }
        
        # Tips based on classification
        tips = {
            "Low": [
                "📖 You can read longer sections at once",
                "⏱️ 45-60 minute study sessions work well",
                "✍️ Light note-taking should be enough"
            ],
            "Medium": [
                "📖 Read in 25-30 minute chunks (Pomodoro style)",
                "✍️ Take notes on key points",
                "🔄 Review what you read before moving on",
                "❓ Write down words you don't know"
            ],
            "High": [
                "📖 Read in 15-20 minute chunks only",
                "☕ Take a 10-minute break between chunks",
                "✍️ Summarize each paragraph in your own words",
                "🎥 Look for YouTube videos explaining the same topic",
                "👥 Discuss with classmates or teachers"
            ]
        }
        
        msg = main_messages.get(classification, main_messages["Medium"])
        
        return {
            "emoji": msg["emoji"],
            "title": msg["title"],
            "summary": msg["summary"],
            "meaning": msg["meaning"],
            "tips": tips.get(classification, tips["Medium"]),
            "score": score,
            "classification": classification
        }

# ============================================================================
# EXAM MODE ANALYZER
# ============================================================================

class ExamModeAnalyzer:
    """Analyzes text to identify high-risk sections for exam preparation."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.extractor = CognitiveFeatureExtractor()
        self.calculator = CognitiveLoadCalculator()
        
    def analyze_sections(self, text: str) -> List[Dict]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        sections = []
        for i, para in enumerate(paragraphs):
            if len(para) < 20:
                continue
            preprocessed = self.preprocessor.full_preprocess(para)
            features = self.extractor.extract_all_features(preprocessed, para)
            score = self.calculator.calculate_score(features)
            classification = self.calculator.classify_load(score)
            sections.append({
                'index': i + 1,
                'text': para[:200] + '...' if len(para) > 200 else para,
                'full_text': para,
                'score': score,
                'classification': classification,
                'word_count': preprocessed['word_count'],
                'is_high_risk': classification == "High" or score > 60
            })
        return sections
    
    def get_high_risk_sections(self, sections: List[Dict]) -> List[Dict]:
        return [s for s in sections if s['is_high_risk']]
    
    def generate_simple_recommendations(self, sections: List[Dict]) -> List[Dict]:
        """Generate simple, actionable recommendations."""
        recommendations = []
        high_risk = self.get_high_risk_sections(sections)
        
        if not high_risk:
            recommendations.append({
                "icon": "✅",
                "title": "Looking Good!",
                "text": "No super-hard sections found. The difficulty is pretty even throughout."
            })
        else:
            recommendations.append({
                "icon": "⚠️",
                "title": f"Found {len(high_risk)} Tricky Part(s)",
                "text": "These sections need extra attention. Don't rush through them!"
            })
            
            for i, section in enumerate(high_risk[:3]):
                recommendations.append({
                    "icon": "📍",
                    "title": f"Section {section['index']} is Hard",
                    "text": f"Difficulty: {section['score']}/100. Read this part slowly and take notes."
                })
        
        avg_score = sum(s['score'] for s in sections) / len(sections) if sections else 0
        
        if avg_score > 70:
            recommendations.append({
                "icon": "💡",
                "title": "Try Other Resources",
                "text": "This text is quite complex. Look for YouTube videos or simpler explanations online."
            })
        elif avg_score > 50:
            recommendations.append({
                "icon": "📝",
                "title": "Make Flashcards",
                "text": "Create flashcards for key terms to help you remember them."
            })
            
        return recommendations

# ============================================================================
# ADAPTIVE CHUNKING SYSTEM
# ============================================================================

class AdaptiveChunker:
    """Provides recommendations for breaking up study sessions."""
    
    def __init__(self):
        self.session_lengths = {
            'Low': (45, 60),
            'Medium': (25, 30),
            'High': (15, 20)
        }
        self.reading_speed = 200
        
    def calculate_study_plan(self, word_count: int, classification: str) -> Dict:
        min_session, max_session = self.session_lengths.get(classification, (25, 30))
        reading_minutes = word_count / self.reading_speed
        
        processing_multiplier = {
            'Low': 1.2,
            'Medium': 1.5,
            'High': 2.0
        }.get(classification, 1.5)
        
        total_study_time = reading_minutes * processing_multiplier
        avg_session = (min_session + max_session) / 2
        num_sessions = max(1, round(total_study_time / avg_session))
        words_per_session = word_count / num_sessions
        
        break_duration = {
            'Low': 5,
            'Medium': 5,
            'High': 10
        }.get(classification, 5)
        
        total_with_breaks = total_study_time + (num_sessions - 1) * break_duration
        
        return {
            'total_words': word_count,
            'reading_time_mins': round(reading_minutes, 1),
            'study_time_mins': round(total_study_time, 1),
            'session_length_min': min_session,
            'session_length_max': max_session,
            'num_sessions': num_sessions,
            'words_per_session': round(words_per_session),
            'break_mins': break_duration,
            'total_time_mins': round(total_with_breaks, 1)
        }
    
    def format_time(self, minutes: float) -> str:
        """Format minutes into a readable string."""
        if minutes < 1:
            return "Less than 1 min"
        elif minutes < 60:
            return f"{int(minutes)} min"
        else:
            hours = int(minutes // 60)
            mins = int(minutes % 60)
            if mins == 0:
                return f"{hours} hr"
            return f"{hours} hr {mins} min"

# ============================================================================
# VISUALIZATION HELPERS (Light Mode)
# ============================================================================

def create_gauge_chart(score: float, classification: str) -> go.Figure:
    """Create a clean, light-mode gauge chart."""
    color = COLORS.get(classification.lower(), COLORS['primary'])
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': '/100', 'font': {'size': 40, 'color': COLORS['text']}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Difficulty Score", 'font': {'size': 20, 'color': COLORS['text']}},
        gauge={
            'axis': {
                'range': [0, 100], 
                'tickwidth': 2, 
                'tickcolor': COLORS['text_secondary'],
                'tickfont': {'color': COLORS['text_secondary']}
            },
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': COLORS['surface'],
            'borderwidth': 0,
            'steps': [
                {'range': [0, LOAD_THRESHOLDS['low']], 'color': '#dcfce7'},
                {'range': [LOAD_THRESHOLDS['low'], LOAD_THRESHOLDS['medium']], 'color': '#fef3c7'},
                {'range': [LOAD_THRESHOLDS['medium'], 100], 'color': '#fee2e2'}
            ],
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text'], 'family': 'Inter, system-ui, sans-serif'}
    )
    
    return fig

def create_feature_bar_chart(features: Dict) -> go.Figure:
    """Create a clean horizontal bar chart."""
    feature_names = []
    normalized_values = []
    colors = []
    simple_names = {
        'sentence_length': 'Sentence Length',
        'vocabulary_complexity': 'Hard Words',
        'abstract_density': 'Abstract Ideas',
        'concept_repetition': 'Repetition (Good!)',
        'readability': 'Reading Difficulty'
    }
    
    for feature_name, feature_data in features.items():
        display_name = simple_names.get(feature_name, feature_name.replace('_', ' ').title())
        feature_names.append(display_name)
        normalized_values.append(feature_data['normalized'])
        
        if feature_data['normalized'] < 35:
            colors.append(COLORS['low'])
        elif feature_data['normalized'] < 65:
            colors.append(COLORS['medium'])
        else:
            colors.append(COLORS['high'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=feature_names,
        x=normalized_values,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=[f"{v:.0f}" for v in normalized_values],
        textposition='outside',
        textfont=dict(size=14, color=COLORS['text']),
        hovertemplate='%{y}: %{x:.0f}/100<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis=dict(
            title="Score (0-100)",
            range=[0, 110],
            gridcolor='#e2e8f0',
            zerolinecolor='#e2e8f0',
            tickfont=dict(color=COLORS['text_secondary'])
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=13, color=COLORS['text'])
        ),
        height=300,
        margin=dict(l=130, r=50, t=20, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font={'family': 'Inter, system-ui, sans-serif'}
    )
    
    return fig

def create_section_chart(sections: List[Dict]) -> go.Figure:
    """Create a bar chart for section-by-section analysis."""
    if not sections:
        return None
        
    section_labels = [f"Part {s['index']}" for s in sections]
    scores = [s['score'] for s in sections]
    
    colors = []
    for score in scores:
        if score < LOAD_THRESHOLDS['low']:
            colors.append(COLORS['low'])
        elif score < LOAD_THRESHOLDS['medium']:
            colors.append(COLORS['medium'])
        else:
            colors.append(COLORS['high'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=section_labels,
        y=scores,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.0f}" for s in scores],
        textposition='outside',
        textfont=dict(size=12, color=COLORS['text']),
        hovertemplate='%{x}<br>Difficulty: %{y:.0f}/100<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis=dict(
            title="",
            tickfont=dict(color=COLORS['text_secondary'])
        ),
        yaxis=dict(
            title="Difficulty Score",
            range=[0, 110],
            gridcolor='#e2e8f0',
            tickfont=dict(color=COLORS['text_secondary'])
        ),
        height=280,
        margin=dict(l=60, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, system-ui, sans-serif'}
    )
    
    return fig

# ============================================================================
# TEST EXAMPLES
# ============================================================================

SAMPLE_TEXTS = {
    "📗 Easy Text": """
The sun is a star. It gives us light and heat. Plants need the sun to grow. 
We see the sun in the sky during the day. At night, we see the moon and stars.

The sun is very big. It is much bigger than Earth. The sun is far away from us.
Light from the sun takes about eight minutes to reach Earth.

Many animals wake up when the sun rises. They sleep when it sets. This is 
called a day and night cycle. The cycle helps all living things.
""",
    
    "📙 Medium Text": """
Photosynthesis is the process by which plants convert light energy into chemical 
energy. This process occurs primarily in the leaves, where chlorophyll captures 
sunlight and uses it to transform carbon dioxide and water into glucose and oxygen.

The process can be divided into two stages: the light-dependent reactions and 
the light-independent reactions (Calvin cycle). During the light reactions, 
water molecules are split, releasing oxygen as a byproduct. The energy captured 
is stored in ATP and NADPH molecules.

In the Calvin cycle, these energy carriers are used to convert carbon dioxide 
into glucose through a series of enzymatic reactions. This glucose serves as 
the primary energy source for the plant and forms the base of most food chains.
""",
    
    "📕 Hard Text": """
The epistemological ramifications of quantum mechanical indeterminacy necessitate 
a fundamental reconceptualization of classical causality paradigms. Heisenberg's 
uncertainty principle demonstrates that conjugate variables such as position and 
momentum cannot be simultaneously determined with arbitrary precision, thereby 
imposing intrinsic limitations on our predictive capabilities.

Furthermore, the Copenhagen interpretation's instrumentalist ontology suggests 
that quantum mechanical formalism should be understood as a mathematical apparatus 
for calculating observational probabilities rather than as a representation of 
an underlying objective reality. This anti-realist stance has been challenged by 
alternative interpretations including Bohmian mechanics and Everettian many-worlds 
theory, each proposing different metaphysical frameworks for understanding quantum 
phenomena.

The decoherence program attempts to explain the emergence of classical behavior 
from quantum substrates through environmental entanglement processes, though this 
approach faces difficulties in providing a complete resolution to the measurement 
problem without additional interpretive assumptions.
"""
}

# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================

def main():
    """Main function to run the CognitiveLoad AI Streamlit application."""
    
    # Page Configuration
    st.set_page_config(
        page_title="CognitiveLoad AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize theme in session state
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # Theme-aware CSS
    is_dark = st.session_state.dark_mode
    
    # Define theme colors
    if is_dark:
        theme = {
            'bg_primary': '#0f172a',
            'bg_secondary': '#1e293b',
            'bg_card': '#1e293b',
            'bg_surface': '#334155',
            'text_primary': '#f1f5f9',
            'text_secondary': '#94a3b8',
            'text_muted': '#64748b',
            'border': '#334155',
            'accent': '#3b82f6',
            'accent_secondary': '#8b5cf6',
            'low_bg': '#064e3b',
            'low_text': '#6ee7b7',
            'medium_bg': '#78350f',
            'medium_text': '#fcd34d',
            'high_bg': '#7f1d1d',
            'high_text': '#fca5a5',
            'simplified_bg': '#064e3b',
            'simplified_border': '#059669',
            'simplified_title': '#6ee7b7',
            'tip_bg': '#1e3a5f',
            'tip_border': '#3b82f6',
            'input_bg': '#1e293b',
            'input_border': '#475569',
        }
    else:
        theme = {
            'bg_primary': '#f8fafc',
            'bg_secondary': '#f1f5f9',
            'bg_card': '#ffffff',
            'bg_surface': '#f8fafc',
            'text_primary': '#1e293b',
            'text_secondary': '#475569',
            'text_muted': '#64748b',
            'border': '#e2e8f0',
            'accent': '#3b82f6',
            'accent_secondary': '#8b5cf6',
            'low_bg': '#dcfce7',
            'low_text': '#166534',
            'medium_bg': '#fef3c7',
            'medium_text': '#92400e',
            'high_bg': '#fee2e2',
            'high_text': '#991b1b',
            'simplified_bg': '#ecfdf5',
            'simplified_border': '#bbf7d0',
            'simplified_title': '#166534',
            'tip_bg': '#eff6ff',
            'tip_border': '#bfdbfe',
            'input_bg': '#ffffff',
            'input_border': '#e2e8f0',
        }
    
    # Dynamic CSS based on theme
    st.markdown(f"""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global styles */
        * {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
        }}
        
        /* Main background */
        .stApp {{
            background: linear-gradient(180deg, {theme['bg_primary']} 0%, {theme['bg_secondary']} 100%);
        }}
        
        /* All text visibility */
        .stApp, .stApp p, .stApp span, .stApp label, .stApp div {{
            color: {theme['text_primary']} !important;
        }}
        
        .stMarkdown p, .stMarkdown span, .stMarkdown li {{
            color: {theme['text_primary']} !important;
        }}
        
        /* Header styling */
        .main-header {{
            text-align: center;
            padding: 2rem 0;
            margin-bottom: 1rem;
        }}
        
        .main-header h1 {{
            background: linear-gradient(135deg, {theme['accent']} 0%, {theme['accent_secondary']} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        
        .main-header p {{
            color: {theme['text_secondary']} !important;
            font-size: 1.1rem;
            font-weight: 400;
        }}
        
        /* Card styling */
        .result-card {{
            background: {theme['bg_card']};
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
            margin-bottom: 1rem;
            border: 1px solid {theme['border']};
        }}
        
        .result-card h3 {{
            color: {theme['text_primary']} !important;
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }}
        
        .result-card p, .result-card span {{
            color: {theme['text_primary']} !important;
        }}
        
        /* Score badge styles */
        .score-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1.5rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.1rem;
        }}
        
        .score-badge.low {{
            background: {theme['low_bg']};
            color: {theme['low_text']} !important;
        }}
        
        .score-badge.medium {{
            background: {theme['medium_bg']};
            color: {theme['medium_text']} !important;
        }}
        
        .score-badge.high {{
            background: {theme['high_bg']};
            color: {theme['high_text']} !important;
        }}
        
        /* Study plan card */
        .study-plan-item {{
            background: {theme['bg_surface']};
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-left: 4px solid {theme['accent']};
        }}
        
        .study-plan-item .value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {theme['text_primary']} !important;
        }}
        
        .study-plan-item .label {{
            font-size: 0.875rem;
            color: {theme['text_secondary']} !important;
        }}
        
        /* Tip cards */
        .tip-card {{
            background: {theme['tip_bg']};
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            border: 1px solid {theme['tip_border']};
        }}
        
        .tip-card p, .tip-card span {{
            color: {theme['text_primary']} !important;
        }}
        
        .tip-card .icon {{
            font-size: 1.25rem;
            margin-right: 0.5rem;
        }}
        
        /* Feature explanation */
        .feature-explanation {{
            background: {theme['bg_surface']};
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: {theme['text_secondary']} !important;
        }}
        
        /* Simplified text box */
        .simplified-box {{
            background: {theme['simplified_bg']};
            border-radius: 12px;
            padding: 1.25rem;
            border: 1px solid {theme['simplified_border']};
            margin: 1rem 0;
        }}
        
        .simplified-box h4 {{
            color: {theme['simplified_title']} !important;
            font-size: 1rem;
            margin-bottom: 0.75rem;
        }}
        
        .simplified-box p {{
            color: {theme['text_primary']} !important;
            line-height: 1.7;
        }}
        
        /* Word change badge */
        .word-change {{
            display: inline-flex;
            align-items: center;
            background: {theme['bg_card']};
            border-radius: 6px;
            padding: 0.25rem 0.5rem;
            margin: 0.25rem;
            font-size: 0.8rem;
            border: 1px solid {theme['border']};
        }}
        
        .word-change .old {{
            color: #ef4444 !important;
            text-decoration: line-through;
            margin-right: 0.25rem;
        }}
        
        .word-change .arrow {{
            color: {theme['text_muted']} !important;
            margin: 0 0.25rem;
        }}
        
        .word-change .new {{
            color: #10b981 !important;
            font-weight: 500;
        }}
        
        /* Section risk badge */
        .section-risk {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        
        .section-risk.high {{
            background: {theme['high_bg']};
            color: {theme['high_text']} !important;
        }}
        
        .section-risk.ok {{
            background: {theme['low_bg']};
            color: {theme['low_text']} !important;
        }}
        
        /* Hide Streamlit elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display: none;}}
        
        /* Text area styling */
        .stTextArea textarea {{
            font-size: 15px;
            line-height: 1.7;
            border-radius: 12px;
            border: 2px solid {theme['input_border']};
            padding: 1rem;
            background: {theme['input_bg']};
            color: {theme['text_primary']} !important;
        }}
        
        .stTextArea textarea:focus {{
            border-color: {theme['accent']};
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }}
        
        .stTextArea textarea::placeholder {{
            color: {theme['text_muted']} !important;
        }}
        
        /* Button styling */
        .stButton > button {{
            background: linear-gradient(135deg, {theme['accent']} 0%, {theme['accent_secondary']} 100%);
            color: white !important;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.2s;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }}
        
        /* Sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {{
            background: {theme['bg_card']};
        }}
        
        [data-testid="stSidebar"] * {{
            color: {theme['text_primary']} !important;
        }}
        
        /* Metric styling */
        [data-testid="stMetricValue"] {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {theme['text_primary']} !important;
        }}
        
        [data-testid="stMetricLabel"] {{
            color: {theme['text_secondary']} !important;
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background: {theme['bg_surface']};
            border-radius: 8px;
            color: {theme['text_primary']} !important;
        }}
        
        /* Radio buttons and checkboxes */
        .stRadio label, .stCheckbox label {{
            color: {theme['text_primary']} !important;
        }}
        
        /* Selectbox */
        .stSelectbox > div > div {{
            background: {theme['input_bg']};
            color: {theme['text_primary']} !important;
        }}
        
        /* Warning, info, success, error boxes */
        .stAlert {{
            color: {theme['text_primary']} !important;
        }}
        
        /* Charts - make them visible */
        .js-plotly-plot .plotly .main-svg {{
            background: transparent !important;
        }}
        
        /* Dataframe styling */
        .stDataFrame {{
            background: {theme['bg_card']};
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {theme['text_primary']} !important;
        }}
        
        /* Theme toggle button */
        .theme-toggle {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            background: {theme['bg_surface']};
            border: 1px solid {theme['border']};
            color: {theme['text_primary']} !important;
            cursor: pointer;
            font-weight: 500;
            margin-bottom: 1rem;
        }}
        
        /* Interpretation text */
        .interpretation-text {{
            color: {theme['text_primary']} !important;
            line-height: 1.8;
            font-size: 1rem;
        }}
        
        /* Section analysis */
        .section-box {{
            background: {theme['bg_surface']};
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border: 1px solid {theme['border']};
        }}
        
        .section-box p, .section-box span {{
            color: {theme['text_primary']} !important;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 CognitiveLoad AI</h1>
        <p>Understand how hard your study material is — and how to tackle it</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Theme Toggle
        theme_col1, theme_col2 = st.columns([1, 1])
        with theme_col1:
            if st.button("☀️ Light" if is_dark else "🌙 Dark", key="theme_toggle", use_container_width=True):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()
        with theme_col2:
            st.markdown(f"<span style='color: {theme['text_secondary']}; font-size: 0.85rem;'>{'Dark' if is_dark else 'Light'} mode</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        mode = st.radio(
            "Analysis Mode",
            ["📊 Standard", "🎓 Exam Mode"],
            help="Exam Mode shows you which parts are hardest"
        )
        
        show_simplified = st.checkbox(
            "✨ Simplify the Text",
            value=True,
            help="Show an easier-to-read version of your text"
        )
        
        st.markdown("---")
        
        st.markdown("### 📝 Try an Example")
        sample_choice = st.selectbox(
            "Load sample text",
            ["-- Pick one --"] + list(SAMPLE_TEXTS.keys())
        )
        
        st.markdown("---")
        
        show_debug = st.checkbox(
            "🔧 Show Technical Details",
            value=False,
            help="For curious minds who want to see how it works"
        )
        
        st.markdown("---")
        
        with st.expander("ℹ️ About"):
            st.markdown("""
            **CognitiveLoad AI** helps you understand how hard your study material is.
            
            **What we check:**
            - 📏 Sentence length
            - 📚 Hard words
            - 💭 Abstract ideas
            - 🔄 Repetition
            - 📖 Overall readability
            
            **Your privacy:** We don't store anything. Everything happens in your browser.
            """)
    
    # Main Content
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    if sample_choice != "-- Pick one --":
        st.session_state.text_input = SAMPLE_TEXTS[sample_choice]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📄 Paste Your Study Material")
        text_input = st.text_area(
            "Enter text to analyze:",
            value=st.session_state.text_input,
            height=280,
            placeholder="Paste your textbook content, lecture notes, or any study material here...\n\nTip: More text = better analysis!",
            key="main_text_input",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        if text_input:
            word_count = len(text_input.split())
            sentence_count = len(sent_tokenize(text_input))
            
            st.metric("Words", f"{word_count:,}")
            st.metric("Sentences", f"{sentence_count:,}")
            
            if word_count < 30:
                st.warning("⚠️ Add more text for better analysis")
            elif word_count < 100:
                st.info("💡 Good start! More text = better results")
            else:
                st.success("✅ Great amount of text!")
        else:
            st.info("👆 Paste text to see stats")
    
    # Analyze Button
    analyze_clicked = st.button(
        "🔍 Analyze My Text",
        type="primary",
        use_container_width=True
    )
    
    # Analysis Results
    if analyze_clicked and text_input:
        if len(text_input.split()) < 10:
            st.error("⚠️ Please add more text (at least 10 words)")
        else:
            with st.spinner("🔍 Analyzing your text..."):
                # Initialize components
                preprocessor = TextPreprocessor()
                extractor = CognitiveFeatureExtractor()
                calculator = CognitiveLoadCalculator()
                chunker = AdaptiveChunker()
                simplifier = TextSimplifier()
                
                # Process
                preprocessed = preprocessor.full_preprocess(text_input)
                features = extractor.extract_all_features(preprocessed, text_input)
                score = calculator.calculate_score(features)
                classification = calculator.classify_load(score)
                interpretation = calculator.get_simple_interpretation(score, classification, features)
                study_plan = chunker.calculate_study_plan(preprocessed['word_count'], classification)
                
                # Simplify text if enabled
                if show_simplified:
                    simplified_result = simplifier.simplify_text(text_input)
            
            st.markdown("---")
            
            # Results Section
            st.markdown("## 📊 Your Results")
            
            col1, col2, col3 = st.columns([1.2, 1, 1])
            
            with col1:
                gauge_fig = create_gauge_chart(score, classification)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                st.markdown("### How Hard Is It?")
                badge_class = classification.lower()
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem;">
                    <div class="score-badge {badge_class}">
                        {interpretation['emoji']} {interpretation['title']}
                    </div>
                    <p style="margin-top: 1rem; color: #64748b; font-size: 0.95rem;">
                        {interpretation['summary']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("### What This Means")
                st.markdown(f"""
                <div style="padding: 0.5rem;">
                    <p style="color: #475569; line-height: 1.7;">
                        {interpretation['meaning']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Study Tips
            st.markdown("## 💡 Study Tips For You")
            
            tip_cols = st.columns(len(interpretation['tips']))
            for i, tip in enumerate(interpretation['tips']):
                with tip_cols[i]:
                    st.markdown(f"""
                    <div class="tip-card">
                        <span style="font-size: 1rem;">{tip}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Adaptive Study Plan (Improved)
            st.markdown("## 📚 Your Study Plan")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ⏱️ Time You'll Need")
                
                st.markdown(f"""
                <div class="study-plan-item">
                    <div class="value">{chunker.format_time(study_plan['reading_time_mins'])}</div>
                    <div class="label">Just reading through it</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="study-plan-item" style="border-left-color: #8b5cf6;">
                    <div class="value">{chunker.format_time(study_plan['study_time_mins'])}</div>
                    <div class="label">To actually learn it (with thinking time)</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="study-plan-item" style="border-left-color: #10b981;">
                    <div class="value">{chunker.format_time(study_plan['total_time_mins'])}</div>
                    <div class="label">Total with breaks included</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📋 How to Break It Up")
                
                st.markdown(f"""
                <div class="result-card">
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; text-align: center;">
                        <div>
                            <div style="font-size: 2rem; font-weight: 700; color: #3b82f6;">
                                {study_plan['num_sessions']}
                            </div>
                            <div style="color: #64748b; font-size: 0.9rem;">study sessions</div>
                        </div>
                        <div>
                            <div style="font-size: 2rem; font-weight: 700; color: #8b5cf6;">
                                {study_plan['session_length_min']}-{study_plan['session_length_max']}
                            </div>
                            <div style="color: #64748b; font-size: 0.9rem;">minutes each</div>
                        </div>
                        <div>
                            <div style="font-size: 2rem; font-weight: 700; color: #10b981;">
                                ~{study_plan['words_per_session']}
                            </div>
                            <div style="color: #64748b; font-size: 0.9rem;">words per session</div>
                        </div>
                        <div>
                            <div style="font-size: 2rem; font-weight: 700; color: #f59e0b;">
                                {study_plan['break_mins']}
                            </div>
                            <div style="color: #64748b; font-size: 0.9rem;">min breaks</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Simple explanation
                if classification == "Low":
                    st.info("😊 This is easy stuff! You can study for longer periods without getting tired.")
                elif classification == "Medium":
                    st.warning("📝 Medium difficulty - use the Pomodoro technique (25 min work, 5 min break)")
                else:
                    st.error("🧠 This is tough! Short sessions with good breaks will help you absorb it better.")
            
            st.markdown("---")
            
            # Feature Breakdown
            st.markdown("## 📈 What Makes It Hard (or Easy)")
            
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                bar_fig = create_feature_bar_chart(features)
                st.plotly_chart(bar_fig, use_container_width=True)
            
            with col2:
                st.markdown("### Feature Explanations")
                for feature_name, feature_data in features.items():
                    with st.expander(f"{feature_data['simple_explanation'][:2]} {feature_name.replace('_', ' ').title()}"):
                        st.markdown(f"**Score:** {feature_data['normalized']:.0f}/100")
                        st.markdown(f"**What it means:** {feature_data['description']}")
                        st.markdown(f"""
                        <div class="feature-explanation">
                            {feature_data['simple_explanation']}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Simplified Text Section
            if show_simplified:
                st.markdown("---")
                st.markdown("## ✨ Simplified Version")
                
                if simplified_result['changes']:
                    st.markdown(f"""
                    <div class="simplified-box">
                        <h4>📖 Here's an easier way to read this:</h4>
                        <p>{simplified_result['simplified']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if simplified_result['word_changes_count'] > 0:
                        st.markdown("**Words we simplified:**")
                        changes_html = ""
                        for change in simplified_result['changes'][:10]:
                            if 'complex word' in change.get('reason', ''):
                                changes_html += f"""
                                <span class="word-change">
                                    <span class="old">{change['original']}</span>
                                    <span class="arrow">→</span>
                                    <span class="new">{change['simplified']}</span>
                                </span>
                                """
                        st.markdown(changes_html, unsafe_allow_html=True)
                else:
                    st.success("✅ This text is already pretty simple! No major changes needed.")
            
            # Exam Mode
            if "Exam" in mode:
                st.markdown("---")
                st.markdown("## 🎓 Exam Mode: Section Analysis")
                
                exam_analyzer = ExamModeAnalyzer()
                sections = exam_analyzer.analyze_sections(text_input)
                
                if sections:
                    section_fig = create_section_chart(sections)
                    if section_fig:
                        st.plotly_chart(section_fig, use_container_width=True)
                    
                    recommendations = exam_analyzer.generate_simple_recommendations(sections)
                    
                    st.markdown("### 📋 What to Focus On")
                    for rec in recommendations:
                        st.markdown(f"""
                        <div class="tip-card">
                            <span class="icon">{rec['icon']}</span>
                            <strong>{rec['title']}</strong><br>
                            <span style="color: #64748b;">{rec['text']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show section details
                    with st.expander("📑 See All Sections"):
                        for section in sections:
                            risk_class = "high" if section['is_high_risk'] else "ok"
                            risk_text = "⚠️ Needs attention" if section['is_high_risk'] else "✅ Manageable"
                            st.markdown(f"""
                            <div class="result-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <strong>Part {section['index']}</strong>
                                    <span class="section-risk {risk_class}">{risk_text}</span>
                                </div>
                                <p style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">
                                    {section['text']}
                                </p>
                                <div style="margin-top: 0.5rem; color: #94a3b8; font-size: 0.8rem;">
                                    {section['word_count']} words • Difficulty: {section['score']}/100
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("📝 Add paragraph breaks (blank lines) in your text for section analysis.")
            
            # Debug Section
            if show_debug:
                st.markdown("---")
                st.markdown("## 🔧 Technical Details")
                
                with st.expander("📊 Processing Stats"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Sentences", preprocessed['sentence_count'])
                        st.metric("Total Words", preprocessed['word_count'])
                        st.metric("Unique Words", preprocessed['unique_words'])
                    with col2:
                        st.metric("Content Words", len(preprocessed['tokens_no_stopwords']))
                        st.metric("Lemmas", len(preprocessed['lemmas']))
                        vocab_richness = (preprocessed['unique_words'] / preprocessed['word_count'] * 100) if preprocessed['word_count'] > 0 else 0
                        st.metric("Vocabulary Richness", f"{vocab_richness:.1f}%")
                
                with st.expander("📐 Raw Feature Values"):
                    feature_df = pd.DataFrame([
                        {
                            'Feature': name.replace('_', ' ').title(),
                            'Raw Value': f"{data['raw']} {data['unit']}",
                            'Score (0-100)': data['normalized'],
                            'Weight': f"{FEATURE_WEIGHTS.get(name, 0)*100:.0f}%"
                        }
                        for name, data in features.items()
                    ])
                    st.dataframe(feature_df, use_container_width=True)
    
    elif analyze_clicked and not text_input:
        st.warning("👆 Please paste some text first!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; padding: 1rem;">
        <p><strong>CognitiveLoad AI</strong> • Helping students learn smarter</p>
        <p style="font-size: 0.8rem;">
            🔒 Your privacy matters: We don't store or share any of your text
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
