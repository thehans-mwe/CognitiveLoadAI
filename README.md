# 🧠 CognitiveLoad AI

**Predict student cognitive overload from educational text**

A modern web application that analyzes study materials and provides actionable insights to help students learn more effectively.

## ✨ Features

- **Cognitive Load Score** (0-100) - Understand how challenging your text is
- **Feature Analysis** - See what makes text difficult (sentence length, vocabulary, abstract concepts)
- **Study Plan** - Get personalized recommendations for breaking up study sessions
- **Section Analysis** - Identify the hardest parts of your material
- **Text Simplification** - Complex words automatically replaced with simpler alternatives
- **Dark/Light Mode** - Easy on your eyes, day or night

## 🚀 Live Demo

Deploy to Vercel and access your app at your custom URL.

## 📁 Project Structure

```
CognitiveLoadAI/
├── api/
│   └── analyze.py    # Serverless Python API
├── index.html        # Frontend (HTML/CSS/JS)
├── vercel.json       # Vercel configuration
├── requirements.txt  # Python dependencies
└── README.md
```

## 🛠 Tech Stack

- **Frontend**: Vanilla HTML, CSS, JavaScript
- **Backend**: Python serverless function
- **Deployment**: Vercel
- **No external dependencies** - Pure Python NLP

## 📊 How It Works

1. **Text Input** - Paste your study material
2. **NLP Analysis** - Text is processed for:
   - Sentence length distribution
   - Vocabulary complexity (rare word ratio)
   - Abstract noun density
   - Concept repetition frequency
   - Readability score (Flesch-based)
3. **Score Calculation** - Weighted combination of features
4. **Recommendations** - Personalized study tips and chunking advice


## 🔒 Privacy

- **No data stored** - All analysis happens in real-time
- **No tracking** - We don't collect personal information
- **Open source** - See exactly how it works

## 📝 API Usage

```javascript
// POST /api/analyze
fetch('/api/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Your study material here...' })
})
```

**Response:**
```json
{
  "score": 65.2,
  "classification": "Medium",
  "features": { ... },
  "study_plan": { ... },
  "tips": [ ... ],
  "sections": [ ... ]
}
```

## 🎓 For Students

CognitiveLoad AI helps you:
- Know what you're getting into before studying
- Plan realistic study sessions
- Identify tough sections to tackle first
- Understand why material feels hard

## 📄 License

MIT License - Use freely for educational purposes.

---

Built with ❤️ for students everywhere.
