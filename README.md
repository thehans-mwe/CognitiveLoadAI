# 🧠 CognitiveLoad AI

**Intelligent Analysis of Educational Text Complexity**

CognitiveLoad AI is a Streamlit-powered web application that helps students understand and manage the cognitive demands of their study materials. By analyzing educational text using advanced NLP techniques, it provides actionable insights to optimize learning efficiency.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage](#-usage)
- [Feature Explanations](#-feature-explanations)
- [Understanding the Score](#-understanding-the-score)
- [Technical Architecture](#-technical-architecture)
- [Design Philosophy](#-design-philosophy)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### Core Analysis
- **📊 Cognitive Load Score (0-100)**: Quantified measure of text complexity
- **🏷️ Load Classification**: Low / Medium / High categorization
- **📈 Multi-feature Analysis**: Five key cognitive load indicators
- **📉 Visual Dashboard**: Interactive charts and visualizations

### Advanced Features
- **🎓 Exam Mode**: Section-by-section analysis highlighting high-risk areas
- **📚 Adaptive Chunking**: Personalized study session recommendations
- **⏱️ Time Estimates**: Realistic reading and study time calculations
- **💡 Transparent AI**: Clear explanations of all calculations

### NLP Processing
- ✅ Tokenization (word & sentence level)
- ✅ Stopword removal
- ✅ Lemmatization
- ✅ Part-of-Speech tagging
- ✅ Readability analysis

---

## 🔬 How It Works

### Processing Pipeline

```
┌─────────────────┐
│  Input Text     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ ──► Tokenization, Stopwords, Lemmatization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │ ──► 5 Cognitive Load Features
│  Extraction     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Score          │ ──► Weighted Calculation (0-100)
│  Calculation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Results &      │ ──► Dashboard, Recommendations
│  Visualization  │
└─────────────────┘
```

### Features Extracted

| Feature | What It Measures | Impact on Load |
|---------|------------------|----------------|
| Sentence Length | Words per sentence | ↑ Longer = ↑ Load |
| Vocabulary Complexity | Rare/technical word ratio | ↑ Complex = ↑ Load |
| Abstract Density | Abstract noun concentration | ↑ Abstract = ↑ Load |
| Concept Repetition | Key concept frequency | ↑ Repetition = ↓ Load |
| Readability | Flesch-Kincaid score | ↓ Readable = ↑ Load |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Quick Start

1. **Clone or Download the Project**
   ```bash
   cd CognitiveLoadAI
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

5. **Open in Browser**
   
   The app will automatically open at `http://localhost:8501`

### First Run Note
On first run, the app will download required NLTK data (punkt, stopwords, wordnet). This happens automatically and only once.

---

## 📖 Usage

### Basic Analysis

1. **Paste Your Text**: Copy educational material into the text area
2. **Click "Analyze"**: Wait for processing (typically 1-3 seconds)
3. **Review Results**: Examine your cognitive load score and breakdown

### Using Sample Texts

The app includes three sample texts for testing:
- **Easy (Low Load)**: Simple text about the sun
- **Medium (Moderate Load)**: Explanation of photosynthesis
- **Hard (High Load)**: Complex quantum mechanics text

### Exam Mode

Enable "Exam Mode" in the sidebar to:
- See section-by-section analysis
- Identify high-risk paragraphs
- Get targeted study recommendations

### Processing Details

Toggle "Show Processing Details" to see:
- Intermediate token values
- POS tag distributions
- Raw feature calculations
- Score computation breakdown

---

## 📊 Feature Explanations

### 1. Average Sentence Length
**What**: Number of words per sentence  
**Why**: Longer sentences require more working memory to process  
**Scale**: 10 words = low load, 30+ words = high load

```python
# Example calculation
sentence = "The quick brown fox jumps over the lazy dog."
length = 9  # words
normalized = (9 - 10) * 5 = 0  # Low load
```

### 2. Vocabulary Complexity
**What**: Ratio of rare/technical words to common words  
**Why**: Unfamiliar terms require more cognitive effort to decode  
**Scale**: 0% rare = 0, 50%+ rare = 100

```python
# Words checked against common word list
# Technical patterns like "-ization", "-ological" add weight
```

### 3. Abstract Noun Density
**What**: Concentration of abstract concepts  
**Why**: Abstract ideas are harder to visualize than concrete ones  
**Detected by**: Suffix patterns (-tion, -ness, -ity, etc.)

### 4. Concept Repetition
**What**: How often key concepts repeat  
**Why**: Repetition aids learning and reduces load  
**Note**: This feature has a NEGATIVE weight (more repetition = lower load)

### 5. Readability Index
**What**: Flesch-Kincaid Reading Ease score  
**Why**: Standard measure of text accessibility  
**Formula**: `206.835 - 1.015(words/sentence) - 84.6(syllables/word)`

| Score | Level | Grade |
|-------|-------|-------|
| 90-100 | Very Easy | 5th grade |
| 60-70 | Standard | 8th-9th grade |
| 30-50 | Difficult | College |
| 0-30 | Very Difficult | Graduate |

---

## 📈 Understanding the Score

### Score Calculation

```
Cognitive Load Score = Σ (feature_normalized × weight)

Weights:
- Sentence Length:       20%
- Vocabulary Complexity: 25%
- Abstract Density:      20%
- Concept Repetition:   -10% (negative = helps reduce load)
- Readability:           25%
```

### Classification Thresholds

| Score | Classification | Meaning |
|-------|---------------|---------|
| 0-35 | **Low** 🟢 | Easy to process, longer study sessions OK |
| 36-65 | **Medium** 🟡 | Moderate effort, use Pomodoro technique |
| 66-100 | **High** 🔴 | Challenging, short sessions with breaks |

### Study Recommendations

| Load Level | Session Length | Break Duration |
|------------|---------------|----------------|
| Low | 45-60 minutes | 5 minutes |
| Medium | 25-30 minutes | 5 minutes |
| High | 15-20 minutes | 10 minutes |

---

## 🏗️ Technical Architecture

### Project Structure

```
CognitiveLoadAI/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Class Hierarchy

```python
TextPreprocessor          # NLP preprocessing pipeline
├── tokenize_sentences()
├── tokenize_words()
├── remove_stopwords()
├── lemmatize()
└── get_pos_tags()

CognitiveFeatureExtractor # Feature extraction
├── calculate_avg_sentence_length()
├── calculate_vocabulary_complexity()
├── calculate_abstract_density()
├── calculate_concept_repetition()
└── calculate_readability()

CognitiveLoadCalculator   # Score computation
├── calculate_score()
├── classify_load()
└── get_interpretation()

ExamModeAnalyzer          # Section analysis
├── analyze_sections()
├── get_high_risk_sections()
└── generate_exam_recommendations()

AdaptiveChunker           # Study planning
├── calculate_study_plan()
├── find_pause_points()
└── get_chunked_text()
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| Streamlit | Web interface |
| NLTK | NLP processing |
| Pandas | Data manipulation |
| Plotly | Interactive visualizations |
| Scikit-learn | Future ML model support |

---

## 🎯 Design Philosophy

### Ethical AI Principles

1. **Transparency**: All calculations are explainable
2. **Privacy**: No data storage or transmission
3. **Educational Focus**: Helps learning, not cheating
4. **Accessibility**: Free, online, no special hardware

### Cognitive Load Theory Basis

This app is grounded in Cognitive Load Theory (Sweller, 1988):
- **Intrinsic Load**: Inherent difficulty of material
- **Extraneous Load**: Poor presentation adding unnecessary burden
- **Germane Load**: Productive effort toward learning

Our features primarily measure intrinsic and extraneous load factors.

### Why These Features?

| Feature | Cognitive Theory Connection |
|---------|---------------------------|
| Sentence Length | Working memory capacity limits |
| Vocabulary | Schema activation and encoding |
| Abstract Density | Dual coding theory (imagery) |
| Repetition | Spaced repetition benefits |
| Readability | Processing fluency |

---

## 🔧 API Reference

### TextPreprocessor

```python
preprocessor = TextPreprocessor()
result = preprocessor.full_preprocess(text)

# Returns:
{
    'sentences': List[str],
    'tokens': List[str],
    'tokens_no_stopwords': List[str],
    'lemmas': List[str],
    'pos_tags': List[Tuple[str, str]],
    'word_count': int,
    'sentence_count': int,
    'unique_words': int
}
```

### CognitiveFeatureExtractor

```python
extractor = CognitiveFeatureExtractor()
features = extractor.extract_all_features(preprocessed, original_text)

# Returns dict with each feature containing:
{
    'feature_name': {
        'raw': float,
        'normalized': float,  # 0-100
        'unit': str,
        'description': str
    }
}
```

### CognitiveLoadCalculator

```python
calculator = CognitiveLoadCalculator()
score = calculator.calculate_score(features)  # 0-100
classification = calculator.classify_load(score)  # Low/Medium/High
```

---

## 🧪 Testing

### Built-in Test Examples

The app includes three test texts:
1. **Easy**: Elementary school level text
2. **Medium**: High school biology content
3. **Hard**: Graduate-level physics

### Expected Results

| Sample | Expected Score | Classification |
|--------|---------------|----------------|
| Easy | 15-25 | Low |
| Medium | 40-55 | Medium |
| Hard | 75-90 | High |

### Running Manual Tests

```python
# In Python console
from app import TextPreprocessor, CognitiveFeatureExtractor, CognitiveLoadCalculator

text = "Your test text here..."
preprocessor = TextPreprocessor()
extractor = CognitiveFeatureExtractor()
calculator = CognitiveLoadCalculator()

preprocessed = preprocessor.full_preprocess(text)
features = extractor.extract_all_features(preprocessed, text)
score = calculator.calculate_score(features)

print(f"Score: {score}")
print(f"Features: {features}")
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **ML Model Integration**: Train regression model on labeled data
2. **More Languages**: Extend beyond English
3. **Domain-Specific Weights**: Adjust for STEM vs. humanities
4. **Accessibility**: Screen reader support
5. **Export Features**: PDF reports, study schedules

---

## 📄 License

MIT License - Feel free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- **Cognitive Load Theory**: John Sweller
- **Flesch-Kincaid**: Rudolf Flesch, J. Peter Kincaid
- **NLTK**: Bird, Klein, and Loper
- **Streamlit**: Streamlit Inc.

---

<div align="center">

**Built with ❤️ for Students**

*CognitiveLoad AI - Making Learning Easier*

</div>
