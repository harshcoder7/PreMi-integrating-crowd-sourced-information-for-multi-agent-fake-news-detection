# Fake News Detection System

A machine learning-based system for detecting fake news articles using Natural Language Processing (NLP) techniques and ensemble learning methods. This project combines traditional ML models with LLM-based approaches for enhanced accuracy.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Collection](#1-data-collection)
  - [2. Dataset Preparation](#2-dataset-preparation)
  - [3. Model Training](#3-model-training)
  - [4. Model Evaluation](#4-model-evaluation)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Data Collection Pipeline**: Automated data collection and preprocessing
- **Balanced Dataset**: LLM-augmented balanced dataset generation
- **Machine Learning Models**: TF-IDF vectorization with Logistic Regression
- **GAN-LLM Integration**: Advanced fake news generation and detection
- **Model Evaluation**: Comprehensive evaluation metrics using Opik framework
- **Text Preprocessing**: NLTK-based text cleaning and normalization

## 📁 Project Structure

```
fake-news-detection/
│
├── Data_Collection.ipynb           # Data collection and initial processing
├── Balanced_Dataset_with_LLM.ipynb # Dataset balancing using LLM
├── gan-llm-implementation.ipynb    # GAN-LLM model implementation
├── evaluation.ipynb                # Model evaluation and metrics
├── crowd_sourced_balanced_dataset.csv  # Balanced training dataset
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## 🔧 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Jupyter Notebook or JupyterLab
- Git

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install the following packages:

```bash
pip install pandas numpy scikit-learn nltk opik llama-index openai python-dotenv nest-asyncio jupyter ipywidgets
```

### Step 4: Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Step 5: Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 Usage

Follow these steps in sequence to set up and run the complete fake news detection system:

### 1. Data Collection

Start by collecting and preprocessing the initial dataset.

```bash
jupyter notebook Data_Collection.ipynb
```

**What it does:**
- Collects fake news data from various sources
- Performs initial data cleaning and validation
- Exports raw dataset for further processing

**Output:** Raw dataset files ready for balancing

### 2. Dataset Preparation

Balance the dataset using LLM techniques to ensure equal representation of fake and real news.

```bash
jupyter notebook Balanced_Dataset_with_LLM.ipynb
```

**What it does:**
- Analyzes class distribution in the raw dataset
- Uses LLM to generate synthetic samples for minority class
- Creates `crowd_sourced_balanced_dataset.csv`
- Validates data quality and balance

**Output:** `crowd_sourced_balanced_dataset.csv` - A balanced dataset ready for training

### 3. Model Training

Train advanced GAN-LLM models for fake news detection:

```bash
jupyter notebook gan-llm-implementation.ipynb
```

**What it does:**
- Implements GAN architecture for fake news generation
- Trains discriminator for detection
- Combines with LLM features for enhanced accuracy
- Saves trained models for evaluation

### 4. Model Evaluation

Evaluate model performance using comprehensive metrics:

```bash
jupyter notebook evaluation.ipynb
```

**What it does:**
- Loads the trained model
- Runs evaluation on test dataset
- Computes metrics: Accuracy, Precision, Recall, F1-Score
- Generates performance visualizations
- Logs results to Opik dashboard

**Key Metrics Tracked:**
- **Equals**: Exact match accuracy for classification
- **Answer Relevance**: Semantic relevance of predictions
- Model comparison (Llama-4 vs DeepSeek-R1)

## 🧠 Model Architecture

### Text Preprocessing Pipeline

1. **Lowercasing**: Convert all text to lowercase
2. **Special Character Removal**: Remove non-alphabetic characters
3. **Stopword Removal**: Filter common English stopwords
4. **Lemmatization**: Reduce words to base form
5. **Whitespace Normalization**: Clean extra spaces

### Feature Extraction

- **TF-IDF Vectorization**:
  - Max features: 5000
  - Captures term importance across documents
  - Sparse matrix representation for efficiency

### GAN-LLM Models

- **Generator**: Creates synthetic fake news samples
- **Discriminator**: Distinguishes real from fake news
- **LLM Integration**: Leverages Llama-4 and DeepSeek-R1 for contextual understanding
- **Output**: Binary classification (FAKE/REAL)

## 📊 Dataset

### Crowd-Sourced Balanced Dataset

- **File**: `crowd_sourced_balanced_dataset.csv`
- **Structure**:
  - Column 0: News article text
  - Column 1: Label (0 = Real, 1 = Fake)
- **Balance**: Equal distribution of fake and real news
- **Source**: Combination of public datasets and LLM-generated samples

### Data Sources

- Public fake news datasets
- Real news from verified sources
- LLM-augmented synthetic samples for balance


## 📈 Model Performance

Results from evaluation notebook:

| Metric | Score |
|--------|-------|
| Accuracy | 85%+ |
| Precision | ~84% |
| Recall | ~86% |
| F1-Score | ~85% |

*Note: Actual performance may vary based on dataset and configuration*

## 🛠️ Troubleshooting

### Issue: NLTK data not found

**Solution**:
```python
import nltk
nltk.download('all')
```

### Issue: API key errors

**Solution**: Ensure `.env` file exists with valid API keys for OpenAI and Groq.

### Issue: Dataset file not found

**Solution**: Run `Data_Collection.ipynb` and `Balanced_Dataset_with_LLM.ipynb` in sequence to generate the dataset.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- NLTK for natural language processing tools
- Scikit-learn for machine learning algorithms
- Opik for experiment tracking and evaluation
- Llama-Index for LLM integration
- Groq for LLM API access

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact [your-email@example.com]

---

**⭐ If you find this project useful, please consider giving it a star!**
