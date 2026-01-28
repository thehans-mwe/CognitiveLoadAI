"""
CognitiveLoad AI - Student Cognitive Overload Prediction System
================================================================
A Streamlit web application that analyzes educational text to predict
cognitive load and help students optimize their learning experience.

Author: CognitiveLoad AI Team
Version: 1.0.0
License: MIT

Design Philosophy:
- Ethical, transparent AI
- No personal data storage
- Educational focus - helping students learn better
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
# Download required NLTK data (runs once, cached thereafter)

@st.cache_resource
def download_nltk_resources():
    """
    Download all required NLTK resources for NLP processing.
    Uses Streamlit caching to avoid repeated downloads.
    
    Resources needed:
    - punkt: Tokenization
    - punkt_tab: Updated tokenization models
    - stopwords: Common word filtering
    - wordnet: Lemmatization
    - averaged_perceptron_tagger: POS tagging
    - averaged_perceptron_tagger_eng: English POS tagger
    """
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

# Abstract noun suffixes - words ending with these are often abstract concepts
ABSTRACT_SUFFIXES = [
    'tion', 'sion', 'ness', 'ment', 'ity', 'ence', 'ance', 
    'ship', 'dom', 'hood', 'ism', 'ology', 'phy', 'ics'
]

# Technical/academic word patterns (common in educational texts)
TECHNICAL_PATTERNS = [
    r'\b\w+ization\b', r'\b\w+ification\b', r'\b\w+ological\b',
    r'\b\w+ometric\b', r'\b\w+aneous\b', r'\b\w+itious\b'
]

# Feature weights for cognitive load calculation
# These weights are based on cognitive load theory research
FEATURE_WEIGHTS = {
    'sentence_length': 0.20,      # Longer sentences = higher load
    'vocabulary_complexity': 0.25, # Complex vocabulary = higher load
    'abstract_density': 0.20,      # Abstract concepts = higher load
    'concept_repetition': -0.10,   # Repetition helps learning (negative weight)
    'readability': 0.25            # Lower readability = higher load
}

# Cognitive load thresholds
LOAD_THRESHOLDS = {
    'low': 35,      # 0-35: Low cognitive load
    'medium': 65,   # 36-65: Medium cognitive load
    'high': 100     # 66-100: High cognitive load
}

# Color scheme for the application
COLORS = {
    'low': '#28a745',      # Green
    'medium': '#ffc107',   # Yellow/Orange
    'high': '#dc3545',     # Red
    'primary': '#4a90d9',  # Blue
    'secondary': '#6c757d' # Gray
}

# ============================================================================
# NLP PREPROCESSING CLASS
# ============================================================================

class TextPreprocessor:
    """
    Handles all NLP preprocessing tasks for educational text analysis.
    
    This class provides methods for:
    - Tokenization (word and sentence level)
    - Stopword removal
    - Lemmatization
    - POS tagging
    
    All methods are designed to be transparent and explainable.
    """
    
    def __init__(self):
        """Initialize the preprocessor with NLTK components."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into individual sentences.
        
        Args:
            text: Raw input text
            
        Returns:
            List of sentence strings
            
        Example:
            "Hello world. How are you?" -> ["Hello world.", "How are you?"]
        """
        # Clean the text first
        text = self._clean_text(text)
        return sent_tokenize(text)
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Split text into individual word tokens.
        
        Args:
            text: Raw input text
            
        Returns:
            List of word tokens (lowercase)
        """
        text = self._clean_text(text)
        tokens = word_tokenize(text.lower())
        # Filter to only alphabetic tokens
        return [token for token in tokens if token.isalpha()]
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove common English stopwords from token list.
        
        Stopwords are common words like "the", "is", "at" that don't
        contribute much to meaning but add to processing load.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered list without stopwords
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Reduce words to their base/dictionary form.
        
        Lemmatization helps identify concept repetition by normalizing
        word forms (e.g., "running" -> "run", "studies" -> "study").
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def get_pos_tags(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Get Part-of-Speech tags for each token.
        
        POS tags help identify nouns, verbs, adjectives, etc.
        This is useful for finding abstract nouns.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of (word, POS_tag) tuples
        """
        return pos_tag(tokens)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize input text.
        
        - Removes extra whitespace
        - Normalizes line breaks
        - Strips leading/trailing whitespace
        """
        # Replace multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def full_preprocess(self, text: str) -> Dict:
        """
        Perform complete preprocessing pipeline on text.
        
        Returns a dictionary with all preprocessing results for
        transparency and debugging.
        
        Args:
            text: Raw input text
            
        Returns:
            Dictionary containing:
            - sentences: List of sentences
            - tokens: All word tokens
            - tokens_no_stopwords: Tokens without stopwords
            - lemmas: Lemmatized tokens
            - pos_tags: POS-tagged tokens
        """
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
# FEATURE EXTRACTION CLASS
# ============================================================================

class CognitiveFeatureExtractor:
    """
    Extracts cognitive load features from preprocessed text.
    
    Features extracted:
    1. Average Sentence Length - Measures syntactic complexity
    2. Vocabulary Complexity - Ratio of rare/technical words
    3. Abstract Noun Density - Concentration of abstract concepts
    4. Concept Repetition - How often key concepts are repeated
    5. Readability Index - Flesch-Kincaid readability score
    
    Each feature is normalized to a 0-100 scale for consistency.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        # Common English words (simplified frequency list)
        # Words NOT in this list are considered "rare"
        self.common_words = self._load_common_words()
        
    def _load_common_words(self) -> set:
        """
        Load a set of common English words.
        
        In production, this would load from a frequency corpus.
        Here we use NLTK stopwords + common academic words.
        """
        common = set(stopwords.words('english'))
        # Add more common words
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
        """
        Calculate average number of words per sentence.
        
        Cognitive Load Theory: Longer sentences require more working memory
        to process, as readers must hold more information before reaching
        the sentence's conclusion.
        
        Args:
            sentences: List of sentence strings
            
        Returns:
            Tuple of (raw_value, normalized_score)
            - raw_value: Average words per sentence
            - normalized_score: 0-100 scale (higher = more complex)
        """
        if not sentences:
            return 0.0, 0.0
            
        lengths = [len(word_tokenize(sent)) for sent in sentences]
        avg_length = sum(lengths) / len(lengths)
        
        # Normalize: 10 words = low (0), 30+ words = high (100)
        # Based on readability research suggesting 15-20 words is optimal
        normalized = min(100, max(0, (avg_length - 10) * 5))
        
        return round(avg_length, 2), round(normalized, 2)
    
    def calculate_vocabulary_complexity(self, tokens: List[str]) -> Tuple[float, float]:
        """
        Calculate ratio of rare/technical words to total words.
        
        Vocabulary complexity increases cognitive load because readers
        must work harder to decode unfamiliar terms.
        
        Args:
            tokens: List of word tokens (lowercase)
            
        Returns:
            Tuple of (raw_ratio, normalized_score)
        """
        if not tokens:
            return 0.0, 0.0
            
        rare_count = 0
        for token in tokens:
            # Check if word is rare (not in common words)
            if token.lower() not in self.common_words:
                rare_count += 1
            # Check for technical patterns
            for pattern in TECHNICAL_PATTERNS:
                if re.match(pattern, token, re.IGNORECASE):
                    rare_count += 0.5  # Partial additional weight
                    break
                    
        ratio = rare_count / len(tokens)
        
        # Normalize: 0% rare = 0, 50%+ rare = 100
        normalized = min(100, ratio * 200)
        
        return round(ratio * 100, 2), round(normalized, 2)
    
    def calculate_abstract_density(self, tokens: List[str], pos_tags: List[Tuple]) -> Tuple[float, float]:
        """
        Calculate density of abstract nouns in the text.
        
        Abstract concepts (like "democracy", "efficiency", "methodology")
        are harder to visualize and require more cognitive effort than
        concrete nouns (like "dog", "table", "book").
        
        Args:
            tokens: List of word tokens
            pos_tags: List of (word, POS) tuples
            
        Returns:
            Tuple of (raw_density, normalized_score)
        """
        if not tokens:
            return 0.0, 0.0
            
        abstract_count = 0
        
        # Method 1: Check for abstract noun suffixes
        for token in tokens:
            for suffix in ABSTRACT_SUFFIXES:
                if token.lower().endswith(suffix) and len(token) > len(suffix) + 2:
                    abstract_count += 1
                    break
        
        # Method 2: Check POS tags for nouns and evaluate
        nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
        
        density = abstract_count / len(tokens) if tokens else 0
        
        # Normalize: 0% = 0, 20%+ = 100
        normalized = min(100, density * 500)
        
        return round(density * 100, 2), round(normalized, 2)
    
    def calculate_concept_repetition(self, lemmas: List[str]) -> Tuple[float, float]:
        """
        Calculate how frequently key concepts are repeated.
        
        Repetition aids learning by reinforcing concepts. Higher repetition
        of key terms typically REDUCES cognitive load (note the negative
        weight in FEATURE_WEIGHTS).
        
        Args:
            lemmas: List of lemmatized tokens (without stopwords)
            
        Returns:
            Tuple of (repetition_ratio, normalized_score)
        """
        if not lemmas:
            return 0.0, 0.0
            
        word_freq = Counter(lemmas)
        
        # Find words that appear more than once
        repeated = sum(1 for count in word_freq.values() if count > 1)
        total_unique = len(word_freq)
        
        if total_unique == 0:
            return 0.0, 0.0
            
        repetition_ratio = repeated / total_unique
        
        # Normalize: 0% repetition = 0, 50%+ = 100
        # Note: Higher repetition is GOOD for learning
        normalized = min(100, repetition_ratio * 200)
        
        return round(repetition_ratio * 100, 2), round(normalized, 2)
    
    def calculate_readability(self, text: str, sentences: List[str], tokens: List[str]) -> Tuple[float, float]:
        """
        Calculate Flesch-Kincaid Reading Ease score.
        
        The Flesch-Kincaid formula considers:
        - Average sentence length (words per sentence)
        - Average syllables per word
        
        Score interpretation:
        - 90-100: Very Easy (5th grade)
        - 60-70: Standard (8th-9th grade)
        - 30-50: Difficult (College)
        - 0-30: Very Difficult (College graduate)
        
        Args:
            text: Original text
            sentences: List of sentences
            tokens: List of word tokens
            
        Returns:
            Tuple of (flesch_score, normalized_cognitive_load)
        """
        if not sentences or not tokens:
            return 0.0, 50.0
            
        # Calculate average sentence length
        avg_sentence_length = len(tokens) / len(sentences)
        
        # Calculate average syllables per word
        total_syllables = sum(self._count_syllables(word) for word in tokens)
        avg_syllables = total_syllables / len(tokens) if tokens else 0
        
        # Flesch Reading Ease formula
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        flesch_score = max(0, min(100, flesch_score))
        
        # Convert to cognitive load (invert the score)
        # Lower readability = Higher cognitive load
        cognitive_load = 100 - flesch_score
        
        return round(flesch_score, 2), round(cognitive_load, 2)
    
    def _count_syllables(self, word: str) -> int:
        """
        Estimate syllable count for a word.
        
        Uses a simple vowel-counting heuristic:
        - Count vowel groups
        - Handle silent 'e'
        - Minimum 1 syllable per word
        """
        word = word.lower()
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
            
        # Handle silent 'e'
        if word.endswith('e') and count > 1:
            count -= 1
            
        # Handle special endings
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
            
        return max(1, count)
    
    def extract_all_features(self, preprocessed: Dict, original_text: str) -> Dict:
        """
        Extract all cognitive load features from preprocessed text.
        
        Args:
            preprocessed: Dictionary from TextPreprocessor.full_preprocess()
            original_text: Original input text
            
        Returns:
            Dictionary containing all features with raw and normalized values
        """
        # Calculate each feature
        sent_len_raw, sent_len_norm = self.calculate_avg_sentence_length(
            preprocessed['sentences']
        )
        
        vocab_raw, vocab_norm = self.calculate_vocabulary_complexity(
            preprocessed['tokens']
        )
        
        abstract_raw, abstract_norm = self.calculate_abstract_density(
            preprocessed['tokens'],
            preprocessed['pos_tags']
        )
        
        rep_raw, rep_norm = self.calculate_concept_repetition(
            preprocessed['lemmas']
        )
        
        read_raw, read_norm = self.calculate_readability(
            original_text,
            preprocessed['sentences'],
            preprocessed['tokens']
        )
        
        return {
            'sentence_length': {
                'raw': sent_len_raw,
                'normalized': sent_len_norm,
                'unit': 'words/sentence',
                'description': 'Average number of words per sentence'
            },
            'vocabulary_complexity': {
                'raw': vocab_raw,
                'normalized': vocab_norm,
                'unit': '% rare words',
                'description': 'Percentage of rare/technical vocabulary'
            },
            'abstract_density': {
                'raw': abstract_raw,
                'normalized': abstract_norm,
                'unit': '% abstract nouns',
                'description': 'Density of abstract concepts'
            },
            'concept_repetition': {
                'raw': rep_raw,
                'normalized': rep_norm,
                'unit': '% repeated concepts',
                'description': 'Frequency of concept reinforcement'
            },
            'readability': {
                'raw': read_raw,
                'normalized': read_norm,
                'unit': 'Flesch score',
                'description': 'Text readability (higher = easier)'
            }
        }

# ============================================================================
# COGNITIVE LOAD CALCULATOR
# ============================================================================

class CognitiveLoadCalculator:
    """
    Calculates overall cognitive load score from extracted features.
    
    Uses a weighted scoring approach based on cognitive load theory.
    The weights can be adjusted based on domain-specific requirements.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize calculator with feature weights.
        
        Args:
            weights: Dictionary mapping feature names to weights.
                     Weights should sum to 1.0 for interpretability.
        """
        self.weights = weights or FEATURE_WEIGHTS
        
    def calculate_score(self, features: Dict) -> float:
        """
        Calculate weighted cognitive load score.
        
        Formula:
        score = Σ (feature_normalized × weight)
        
        Note: concept_repetition has negative weight because
        repetition REDUCES cognitive load.
        
        Args:
            features: Dictionary from CognitiveFeatureExtractor
            
        Returns:
            Cognitive load score (0-100)
        """
        score = 0.0
        
        for feature_name, weight in self.weights.items():
            if feature_name in features:
                normalized_value = features[feature_name]['normalized']
                
                # For concept_repetition (negative weight), 
                # higher repetition = lower contribution to load
                if weight < 0:
                    contribution = abs(weight) * (100 - normalized_value)
                else:
                    contribution = weight * normalized_value
                    
                score += contribution
                
        # Ensure score is within bounds
        return round(max(0, min(100, score)), 1)
    
    def classify_load(self, score: float) -> str:
        """
        Classify cognitive load level based on score.
        
        Args:
            score: Cognitive load score (0-100)
            
        Returns:
            Classification string: "Low", "Medium", or "High"
        """
        if score <= LOAD_THRESHOLDS['low']:
            return "Low"
        elif score <= LOAD_THRESHOLDS['medium']:
            return "Medium"
        else:
            return "High"
    
    def get_load_color(self, classification: str) -> str:
        """Get color code for load classification."""
        return COLORS.get(classification.lower(), COLORS['secondary'])
    
    def get_interpretation(self, score: float, classification: str) -> str:
        """
        Generate human-readable interpretation of the score.
        
        Args:
            score: Cognitive load score
            classification: Load classification
            
        Returns:
            Interpretation text for students
        """
        interpretations = {
            "Low": f"""
                **Great news!** This text has a low cognitive load score of {score}/100.
                
                This means:
                - The content is relatively easy to process
                - Sentence structures are manageable
                - Vocabulary is accessible
                - You should be able to study this in longer sessions
                
                💡 **Recommendation:** You can study this material comfortably 
                for 45-60 minutes before taking a break.
            """,
            "Medium": f"""
                **Moderate complexity detected.** This text has a medium cognitive 
                load score of {score}/100.
                
                This means:
                - The content requires focused attention
                - Some complex concepts or vocabulary present
                - Active reading strategies will help
                
                💡 **Recommendation:** Study in 25-30 minute intervals 
                (Pomodoro technique). Take notes to reinforce understanding.
            """,
            "High": f"""
                **Challenging content ahead!** This text has a high cognitive 
                load score of {score}/100.
                
                This means:
                - Dense, complex material requiring significant mental effort
                - May contain technical vocabulary or abstract concepts
                - Multiple readings may be needed
                
                💡 **Recommendation:** Break into 15-20 minute study sessions.
                Use active recall and summarization. Consider finding 
                supplementary resources.
            """
        }
        return interpretations.get(classification, "Unable to interpret score.")

# ============================================================================
# EXAM MODE ANALYZER
# ============================================================================

class ExamModeAnalyzer:
    """
    Analyzes text to identify high-risk sections for exam preparation.
    
    "High-risk" sections are paragraphs or sentences with particularly
    high cognitive load that students should pay extra attention to.
    """
    
    def __init__(self):
        """Initialize the exam mode analyzer."""
        self.preprocessor = TextPreprocessor()
        self.extractor = CognitiveFeatureExtractor()
        self.calculator = CognitiveLoadCalculator()
        
    def analyze_sections(self, text: str) -> List[Dict]:
        """
        Analyze text by sections (paragraphs) and identify high-risk areas.
        
        Args:
            text: Original input text
            
        Returns:
            List of section analysis dictionaries
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            paragraphs = [text]
            
        sections = []
        
        for i, para in enumerate(paragraphs):
            if len(para) < 20:  # Skip very short paragraphs
                continue
                
            # Analyze this paragraph
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
        """Filter to only high-risk sections."""
        return [s for s in sections if s['is_high_risk']]
    
    def generate_exam_recommendations(self, sections: List[Dict]) -> List[str]:
        """
        Generate study recommendations based on section analysis.
        
        Args:
            sections: List of analyzed sections
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        high_risk = self.get_high_risk_sections(sections)
        
        if not high_risk:
            recommendations.append("✅ No high-risk sections identified. "
                                 "The text is relatively uniform in difficulty.")
        else:
            recommendations.append(f"⚠️ Found {len(high_risk)} high-risk section(s) "
                                 "requiring extra attention.")
            
            for section in high_risk[:3]:  # Top 3 most important
                recommendations.append(
                    f"📍 Section {section['index']} (Score: {section['score']}): "
                    f"Consider re-reading and making notes."
                )
                
        # Add general recommendations
        avg_score = sum(s['score'] for s in sections) / len(sections) if sections else 0
        
        if avg_score > 70:
            recommendations.append("📚 Overall high complexity - consider finding "
                                 "supplementary materials or video explanations.")
        elif avg_score > 50:
            recommendations.append("📝 Create flashcards for key concepts to aid retention.")
            
        return recommendations

# ============================================================================
# ADAPTIVE CHUNKING SYSTEM
# ============================================================================

class AdaptiveChunker:
    """
    Provides recommendations for breaking up study sessions based on
    cognitive load analysis.
    
    Uses research-based guidelines:
    - Low load: 45-60 minute sessions
    - Medium load: 25-30 minute sessions (Pomodoro)
    - High load: 15-20 minute sessions
    """
    
    def __init__(self):
        """Initialize the adaptive chunker."""
        self.session_lengths = {
            'Low': (45, 60),
            'Medium': (25, 30),
            'High': (15, 20)
        }
        
        # Average reading speed: ~200-250 words per minute for educational text
        self.reading_speed = 200  # Conservative estimate
        
    def calculate_study_plan(self, word_count: int, classification: str) -> Dict:
        """
        Generate a study plan based on text length and cognitive load.
        
        Args:
            word_count: Total word count of the text
            classification: Cognitive load classification
            
        Returns:
            Dictionary with study plan details
        """
        min_session, max_session = self.session_lengths.get(
            classification, (25, 30)
        )
        
        # Calculate total reading time
        reading_minutes = word_count / self.reading_speed
        
        # Add processing time based on load
        processing_multiplier = {
            'Low': 1.2,
            'Medium': 1.5,
            'High': 2.0
        }.get(classification, 1.5)
        
        total_study_time = reading_minutes * processing_multiplier
        
        # Calculate number of sessions
        avg_session = (min_session + max_session) / 2
        num_sessions = max(1, round(total_study_time / avg_session))
        
        # Calculate words per session
        words_per_session = word_count / num_sessions
        
        # Calculate break durations
        break_duration = {
            'Low': 5,
            'Medium': 5,
            'High': 10
        }.get(classification, 5)
        
        return {
            'total_words': word_count,
            'estimated_reading_time': round(reading_minutes, 1),
            'estimated_study_time': round(total_study_time, 1),
            'recommended_session_length': f"{min_session}-{max_session} minutes",
            'number_of_sessions': num_sessions,
            'words_per_session': round(words_per_session),
            'break_duration': f"{break_duration} minutes",
            'total_with_breaks': round(total_study_time + (num_sessions - 1) * break_duration, 1)
        }
    
    def find_pause_points(self, text: str, num_chunks: int) -> List[int]:
        """
        Find natural pause points in the text for chunking.
        
        Prefers paragraph breaks, then sentence endings.
        
        Args:
            text: Original text
            num_chunks: Desired number of chunks
            
        Returns:
            List of character positions for pause points
        """
        if num_chunks <= 1:
            return []
            
        # Split into paragraphs first
        paragraphs = text.split('\n')
        
        chunk_size = len(text) // num_chunks
        pause_points = []
        current_pos = 0
        
        for i in range(1, num_chunks):
            target_pos = i * chunk_size
            
            # Find nearest paragraph break
            best_break = target_pos
            
            # Look for paragraph breaks within ±20% of target
            search_range = chunk_size // 5
            
            for j in range(target_pos - search_range, target_pos + search_range):
                if 0 < j < len(text):
                    if text[j] == '\n' or (j > 0 and text[j-1:j+1] == '. '):
                        best_break = j
                        break
                        
            pause_points.append(best_break)
            
        return pause_points
    
    def get_chunked_text(self, text: str, pause_points: List[int]) -> List[str]:
        """
        Split text at pause points.
        
        Args:
            text: Original text
            pause_points: List of character positions
            
        Returns:
            List of text chunks
        """
        if not pause_points:
            return [text]
            
        chunks = []
        prev_point = 0
        
        for point in pause_points:
            chunks.append(text[prev_point:point].strip())
            prev_point = point
            
        chunks.append(text[prev_point:].strip())
        
        return [c for c in chunks if c]  # Remove empty chunks

# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def create_gauge_chart(score: float, classification: str) -> go.Figure:
    """
    Create a gauge chart for the cognitive load score.
    
    Args:
        score: Cognitive load score (0-100)
        classification: Load classification
        
    Returns:
        Plotly figure object
    """
    color = COLORS.get(classification.lower(), COLORS['primary'])
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Cognitive Load Score", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': COLORS['high']}, 
               'decreasing': {'color': COLORS['low']}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, LOAD_THRESHOLDS['low']], 'color': '#d4edda'},
                {'range': [LOAD_THRESHOLDS['low'], LOAD_THRESHOLDS['medium']], 
                 'color': '#fff3cd'},
                {'range': [LOAD_THRESHOLDS['medium'], 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#333', 'family': 'Arial'}
    )
    
    return fig

def create_feature_radar_chart(features: Dict) -> go.Figure:
    """
    Create a radar chart showing feature breakdown.
    
    Args:
        features: Dictionary of extracted features
        
    Returns:
        Plotly figure object
    """
    categories = []
    values = []
    
    for feature_name, feature_data in features.items():
        # Clean up feature name for display
        display_name = feature_name.replace('_', ' ').title()
        categories.append(display_name)
        values.append(feature_data['normalized'])
    
    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(74, 144, 217, 0.3)',
        line=dict(color=COLORS['primary'], width=2),
        name='Feature Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            )
        ),
        showlegend=False,
        title=dict(
            text="Feature Analysis Radar",
            font=dict(size=18)
        ),
        height=400,
        margin=dict(l=80, r=80, t=80, b=40),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_feature_bar_chart(features: Dict) -> go.Figure:
    """
    Create a horizontal bar chart of feature values.
    
    Args:
        features: Dictionary of extracted features
        
    Returns:
        Plotly figure object
    """
    feature_names = []
    raw_values = []
    normalized_values = []
    colors = []
    
    for feature_name, feature_data in features.items():
        display_name = feature_name.replace('_', ' ').title()
        feature_names.append(display_name)
        raw_values.append(f"{feature_data['raw']} {feature_data['unit']}")
        normalized_values.append(feature_data['normalized'])
        
        # Color based on normalized value
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
        marker=dict(color=colors),
        text=raw_values,
        textposition='auto',
        hovertemplate='%{y}: %{x:.1f}/100<br>Raw: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Feature Breakdown",
            font=dict(size=18)
        ),
        xaxis=dict(
            title="Normalized Score (0-100)",
            range=[0, 100]
        ),
        yaxis=dict(title=""),
        height=350,
        margin=dict(l=150, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def create_section_heatmap(sections: List[Dict]) -> go.Figure:
    """
    Create a heatmap visualization of section complexity.
    
    Args:
        sections: List of analyzed sections
        
    Returns:
        Plotly figure object
    """
    if not sections:
        return None
        
    section_labels = [f"Section {s['index']}" for s in sections]
    scores = [s['score'] for s in sections]
    
    # Create color scale
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
        marker=dict(color=colors),
        text=[f"{s:.0f}" for s in scores],
        textposition='outside',
        hovertemplate='%{x}<br>Score: %{y:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Section-by-Section Cognitive Load",
            font=dict(size=18)
        ),
        xaxis=dict(title=""),
        yaxis=dict(title="Cognitive Load Score", range=[0, 110]),
        height=300,
        margin=dict(l=60, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ============================================================================
# TEST EXAMPLES
# ============================================================================

SAMPLE_TEXTS = {
    "Easy (Low Load)": """
The sun is a star. It gives us light and heat. Plants need the sun to grow. 
We see the sun in the sky during the day. At night, we see the moon and stars.

The sun is very big. It is much bigger than Earth. The sun is far away from us.
Light from the sun takes about eight minutes to reach Earth.

Many animals wake up when the sun rises. They sleep when it sets. This is 
called a day and night cycle. The cycle helps all living things.
""",
    
    "Medium (Moderate Load)": """
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
    
    "Hard (High Load)": """
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
    """
    Main function to run the CognitiveLoad AI Streamlit application.
    """
    
    # ========================================================================
    # PAGE CONFIGURATION
    # ========================================================================
    
    st.set_page_config(
        page_title="CognitiveLoad AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # ========================================================================
    # CUSTOM CSS STYLING
    # ========================================================================
    
    st.markdown("""
    <style>
        /* Main container styling */
        .main-header {
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 2rem;
        }
        
        .main-header h1 {
            color: #4a90d9;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .main-header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        /* Card styling */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .metric-card h3 {
            margin: 0;
            font-size: 2rem;
        }
        
        .metric-card p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
        
        /* Status badges */
        .status-low {
            background-color: #d4edda;
            color: #155724;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
            font-weight: bold;
        }
        
        .status-medium {
            background-color: #fff3cd;
            color: #856404;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
            font-weight: bold;
        }
        
        .status-high {
            background-color: #f8d7da;
            color: #721c24;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
            font-weight: bold;
        }
        
        /* Info boxes */
        .info-box {
            background-color: #e7f3ff;
            border-left: 4px solid #4a90d9;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }
        
        .warning-box {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }
        
        /* Feature explanation cards */
        .feature-card {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Improve text area */
        .stTextArea textarea {
            font-size: 14px;
            line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    st.markdown("""
    <div class="main-header">
        <h1>🧠 CognitiveLoad AI</h1>
        <p>Intelligent Analysis of Educational Text Complexity</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Mode selection
        mode = st.radio(
            "Analysis Mode",
            ["Standard Analysis", "Exam Mode"],
            help="Exam Mode provides section-by-section analysis and highlights high-risk areas"
        )
        
        st.markdown("---")
        
        # Sample text selection
        st.markdown("### 📝 Sample Texts")
        st.markdown("Try these examples to see how the system works:")
        
        sample_choice = st.selectbox(
            "Load a sample",
            ["-- Select --"] + list(SAMPLE_TEXTS.keys())
        )
        
        st.markdown("---")
        
        # Show debug info toggle
        show_debug = st.checkbox(
            "Show Processing Details",
            value=False,
            help="Display intermediate values and technical details"
        )
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️ About This App"):
            st.markdown("""
            **CognitiveLoad AI** helps students understand the complexity 
            of educational materials.
            
            **How it works:**
            1. Paste your study text
            2. Our NLP engine analyzes multiple features
            3. Get a cognitive load score and recommendations
            
            **Features analyzed:**
            - Sentence complexity
            - Vocabulary difficulty
            - Abstract concept density
            - Concept repetition
            - Overall readability
            
            **Privacy:** No data is stored. All processing 
            happens in your session.
            
            Made with ❤️ for students
            """)
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    # Initialize session state
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    # Handle sample text selection
    if sample_choice != "-- Select --":
        st.session_state.text_input = SAMPLE_TEXTS[sample_choice]
    
    # Text input area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📄 Enter Educational Text")
        text_input = st.text_area(
            "Paste your study material below:",
            value=st.session_state.text_input,
            height=300,
            placeholder="Paste the educational text you want to analyze here...\n\n"
                       "For best results, include at least 50 words.",
            key="main_text_input"
        )
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        if text_input:
            word_count = len(text_input.split())
            char_count = len(text_input)
            para_count = len([p for p in text_input.split('\n\n') if p.strip()])
            
            st.metric("Words", f"{word_count:,}")
            st.metric("Characters", f"{char_count:,}")
            st.metric("Paragraphs", para_count)
            
            if word_count < 50:
                st.warning("⚠️ For accurate analysis, please provide at least 50 words.")
        else:
            st.info("👆 Enter text to see statistics")
    
    # Analyze button
    analyze_clicked = st.button(
        "🔍 Analyze Cognitive Load",
        type="primary",
        use_container_width=True
    )
    
    # ========================================================================
    # ANALYSIS AND RESULTS
    # ========================================================================
    
    if analyze_clicked and text_input:
        if len(text_input.split()) < 10:
            st.error("⚠️ Please enter more text for meaningful analysis (at least 10 words).")
        else:
            # Show loading spinner
            with st.spinner("Analyzing text... This may take a moment."):
                
                # Initialize components
                preprocessor = TextPreprocessor()
                extractor = CognitiveFeatureExtractor()
                calculator = CognitiveLoadCalculator()
                chunker = AdaptiveChunker()
                
                # Perform preprocessing
                preprocessed = preprocessor.full_preprocess(text_input)
                
                # Extract features
                features = extractor.extract_all_features(preprocessed, text_input)
                
                # Calculate score
                score = calculator.calculate_score(features)
                classification = calculator.classify_load(score)
                interpretation = calculator.get_interpretation(score, classification)
                
                # Get study plan
                study_plan = chunker.calculate_study_plan(
                    preprocessed['word_count'], 
                    classification
                )
            
            # ================================================================
            # RESULTS DASHBOARD
            # ================================================================
            
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")
            
            # Main metrics row
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Gauge chart
                gauge_fig = create_gauge_chart(score, classification)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                st.markdown("### Load Classification")
                status_class = f"status-{classification.lower()}"
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem;">
                    <span class="{status_class}" style="font-size: 1.5rem;">
                        {classification} Cognitive Load
                    </span>
                    <p style="margin-top: 1rem; color: #666;">
                        Score: {score}/100
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick interpretation
                if classification == "Low":
                    st.success("✅ This text is relatively easy to process!")
                elif classification == "Medium":
                    st.warning("⚡ Moderate complexity - active reading recommended")
                else:
                    st.error("🔥 High complexity - break into smaller sessions")
            
            with col3:
                st.markdown("### Study Recommendation")
                st.metric(
                    "Suggested Session Length",
                    study_plan['recommended_session_length']
                )
                st.metric(
                    "Number of Sessions",
                    study_plan['number_of_sessions']
                )
                st.metric(
                    "Break Duration",
                    study_plan['break_duration']
                )
            
            st.markdown("---")
            
            # Feature breakdown
            st.markdown("### 📈 Feature Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                bar_fig = create_feature_bar_chart(features)
                st.plotly_chart(bar_fig, use_container_width=True)
            
            with col2:
                radar_fig = create_feature_radar_chart(features)
                st.plotly_chart(radar_fig, use_container_width=True)
            
            # Detailed interpretation
            st.markdown("### 💡 Detailed Interpretation")
            st.markdown(interpretation)
            
            # Feature explanations
            with st.expander("🔍 Understanding Each Feature"):
                for feature_name, feature_data in features.items():
                    display_name = feature_name.replace('_', ' ').title()
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**{display_name}**")
                        st.caption(feature_data['description'])
                    with col2:
                        st.metric("Raw Value", f"{feature_data['raw']} {feature_data['unit']}")
                    with col3:
                        # Color-coded score
                        color = COLORS['low'] if feature_data['normalized'] < 35 else \
                                COLORS['medium'] if feature_data['normalized'] < 65 else \
                                COLORS['high']
                        st.markdown(f"""
                        <div style="background-color: {color}; 
                                    padding: 0.5rem; 
                                    border-radius: 5px; 
                                    text-align: center;
                                    color: white;">
                            <strong>{feature_data['normalized']}/100</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
            
            # ================================================================
            # EXAM MODE
            # ================================================================
            
            if mode == "Exam Mode":
                st.markdown("---")
                st.markdown("## 🎓 Exam Mode Analysis")
                
                exam_analyzer = ExamModeAnalyzer()
                sections = exam_analyzer.analyze_sections(text_input)
                
                if sections:
                    # Section heatmap
                    heatmap_fig = create_section_heatmap(sections)
                    if heatmap_fig:
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    # High-risk sections
                    high_risk = exam_analyzer.get_high_risk_sections(sections)
                    
                    if high_risk:
                        st.markdown("### ⚠️ High-Risk Sections")
                        st.warning(f"Found {len(high_risk)} section(s) with elevated cognitive load")
                        
                        for section in high_risk:
                            with st.expander(f"Section {section['index']} - Score: {section['score']}/100"):
                                st.markdown(f"**Preview:** {section['text']}")
                                st.markdown(f"**Word Count:** {section['word_count']}")
                                st.info("💡 Tip: Read this section slowly and take notes. "
                                       "Consider re-reading after completing the full text.")
                    else:
                        st.success("✅ No high-risk sections identified. "
                                  "Complexity is fairly uniform throughout.")
                    
                    # Exam recommendations
                    st.markdown("### 📋 Study Recommendations")
                    recommendations = exam_analyzer.generate_exam_recommendations(sections)
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                else:
                    st.info("Text is too short for section analysis. "
                           "Try adding more content with paragraph breaks.")
            
            # ================================================================
            # ADAPTIVE CHUNKING
            # ================================================================
            
            st.markdown("---")
            st.markdown("## 📚 Adaptive Study Plan")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ⏱️ Time Estimates")
                
                st.markdown(f"""
                <div class="info-box">
                    <h4>📖 Reading Time</h4>
                    <p><strong>{study_plan['estimated_reading_time']} minutes</strong> 
                       (at ~200 words/min)</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="info-box">
                    <h4>📝 Study Time</h4>
                    <p><strong>{study_plan['estimated_study_time']} minutes</strong> 
                       (including processing time)</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="info-box">
                    <h4>☕ Total with Breaks</h4>
                    <p><strong>{study_plan['total_with_breaks']} minutes</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🎯 Chunking Strategy")
                
                num_sessions = study_plan['number_of_sessions']
                words_per_session = study_plan['words_per_session']
                
                st.markdown(f"""
                Based on the **{classification}** cognitive load, we recommend:
                
                - **{num_sessions} study session(s)**
                - **~{words_per_session} words per session**
                - **{study_plan['break_duration']} breaks** between sessions
                
                **Why this works:**
                """)
                
                if classification == "Low":
                    st.markdown("""
                    - Text is accessible, allowing longer focus periods
                    - Less mental fatigue expected
                    - Efficient for quick comprehension
                    """)
                elif classification == "Medium":
                    st.markdown("""
                    - Balanced complexity requires moderate breaks
                    - Pomodoro technique (25 min) is ideal
                    - Active note-taking recommended
                    """)
                else:
                    st.markdown("""
                    - High complexity demands shorter bursts
                    - Prevents cognitive overload
                    - More frequent review needed
                    - Consider supplementary resources
                    """)
            
            # ================================================================
            # DEBUG / PROCESSING DETAILS
            # ================================================================
            
            if show_debug:
                st.markdown("---")
                st.markdown("## 🔧 Processing Details")
                
                with st.expander("📊 Preprocessing Statistics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Sentences", preprocessed['sentence_count'])
                        st.metric("Total Words", preprocessed['word_count'])
                        st.metric("Unique Words", preprocessed['unique_words'])
                    with col2:
                        st.metric("Tokens (no stopwords)", 
                                 len(preprocessed['tokens_no_stopwords']))
                        st.metric("Lemmas", len(preprocessed['lemmas']))
                        vocab_richness = (preprocessed['unique_words'] / 
                                         preprocessed['word_count'] * 100) \
                                         if preprocessed['word_count'] > 0 else 0
                        st.metric("Vocabulary Richness", f"{vocab_richness:.1f}%")
                
                with st.expander("📝 Sample Processed Tokens"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**First 20 Tokens:**")
                        st.code(preprocessed['tokens'][:20])
                    with col2:
                        st.markdown("**First 20 Lemmas:**")
                        st.code(preprocessed['lemmas'][:20])
                
                with st.expander("🏷️ POS Tags Sample"):
                    st.write(preprocessed['pos_tags'][:30])
                
                with st.expander("📐 Raw Feature Values"):
                    feature_df = pd.DataFrame([
                        {
                            'Feature': name.replace('_', ' ').title(),
                            'Raw Value': f"{data['raw']} {data['unit']}",
                            'Normalized': data['normalized'],
                            'Weight': FEATURE_WEIGHTS.get(name, 0)
                        }
                        for name, data in features.items()
                    ])
                    st.dataframe(feature_df, use_container_width=True)
                
                with st.expander("⚖️ Score Calculation"):
                    st.markdown("**Formula:** `score = Σ (feature_normalized × weight)`")
                    st.markdown("**Weights used:**")
                    for feature, weight in FEATURE_WEIGHTS.items():
                        direction = "reduces" if weight < 0 else "increases"
                        st.markdown(f"- **{feature.replace('_', ' ').title()}:** "
                                   f"{abs(weight)*100:.0f}% weight ({direction} load)")
                    st.markdown(f"**Final Score:** {score}/100")
    
    elif analyze_clicked and not text_input:
        st.warning("⚠️ Please enter some text to analyze.")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>CognitiveLoad AI</strong> | Built for Students, by AI</p>
        <p style="font-size: 0.8rem;">
            🔒 Privacy First: No data is stored or transmitted. All processing happens locally.
            <br>
            📚 Educational Purpose: This tool helps understand text complexity, not circumvent learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
