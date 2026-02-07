# Confluence Bot (GPT)

A chatbot application that uses fine-tuned T5 model and sentence embeddings to answer questions based on Confluence page data.

## Project Overview

This project consists of three main components:
1. **Dataset Generator** - Creates synthetic Confluence page data
2. **T5 Q&A Fine-Tuning** - Trains a T5 model for question answering
3. **Confluence Chatbot** - Interactive GUI application for querying Confluence data

## Prerequisites

### Required Python Packages

```bash
pip install pandas faker numpy transformers datasets torch sentence-transformers scikit-learn pillow
```

### System Requirements
- Python 3.8 or higher
- Windows OS (for tkinter GUI)
- At least 4GB RAM (for model loading)
- CUDA-compatible GPU (optional, for faster training)

## Project Structure

```
Confluence Bot/
├── Dataset_Generator.ipynb          # Generates confluence_pages.csv
├── T5_QnA_FineTuning.ipynb         # Fine-tunes T5 model
├── Confluence_ChatBot.ipynb  # Main chatbot application
├── confluence_pages.csv            # Generated dataset (created in Step 1)
├── trained_t5_model/               # Fine-tuned model (created in Step 2)
├── bot.jpg                         # Chatbot avatar image
└── README.md                       # This file
```

## Step-by-Step Setup and Execution

### Step 1: Generate Dataset

**File:** `Dataset_Generator.ipynb`

**Purpose:** Creates a CSV file with 1000 sample Confluence page records containing titles, content, resources, links, authors, and metadata.

**Instructions:**
1. Open `Dataset_Generator.ipynb` in VS Code or Jupyter
2. Run all cells (Cell → Run All)
3. Verify that `confluence_pages.csv` is created in the workspace folder

**Output:** `confluence_pages.csv` (1000 rows with 8 columns)

**Note:** Skip this step if you already have `confluence_pages.csv` or want to use your own Confluence dataset.

---

### Step 2: Fine-Tune T5 Model

**File:** `T5_QnA_FineTuning.ipynb`

**Purpose:** Fine-tunes a T5-small model on Q&A pairs to improve question answering capabilities.

**Instructions:**
1. Open `T5_QnA_FineTuning.ipynb`
2. Run all cells (Cell → Run All)
3. Wait for training to complete (3 epochs, approximately 5-10 minutes on CPU)
4. Verify that `./trained_t5_model` directory is created with model files

**Output:** `./trained_t5_model/` directory containing:
- `config.json`
- `pytorch_model.bin`
- `tokenizer_config.json`
- Other model files

**Training Details:**
- Base model: T5-small
- Training samples: 10 Q&A pairs
- Evaluation samples: 10 Q&A pairs
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 8

---

### Step 3: Run the Chatbot

**File:** `Confluence_ChatBot.ipynb`

**Purpose:** Launches an interactive GUI chatbot that answers questions based on the Confluence dataset.

**Prerequisites:**
- ✅ `confluence_pages.csv` exists (from Step 1)
- ✅ `./trained_t5_model/` exists (from Step 2)
- ✅ `bot.jpg` exists at: `C:\Users\locperera\OneDrive\Documents\AI Hackathon\bot.jpg`

**Instructions:**
1. Open Confluence_ChatBot.ipynb`
2. Verify all prerequisites are met
3. Run all cells (Cell → Run All)
4. The WileyGPT GUI window will appear
5. Type your question in the input field and click "Submit"

**Features:**
- Semantic search using sentence transformers
- T5-based answer generation
- Similarity scoring for relevance
- Scrollable conversation history
- Modern GUI with custom branding

---

## How It Works

### Architecture

1. **Embedding Model:** Uses `paraphrase-MiniLM-L6-v2` to create embeddings for Confluence pages
2. **Retrieval:** Cosine similarity matches user queries to the most relevant Confluence page
3. **Answer Generation:** Fine-tuned T5 model generates answers based on the retrieved context
4. **GUI:** Tkinter-based interface for user interaction

### Workflow

```
User Query → Embedding → Similarity Search → Retrieve Context → T5 Model → Answer
```

## Usage Examples

Once the chatbot is running, you can ask questions like:
- "What is Machine Learning?"
- "How does Power BI help in data analysis?"
- "What is the difference between supervised and unsupervised learning?"
- "How to set up an ETL pipeline?"
- "What are best practices for data warehousing?"

## Troubleshooting

### Issue: `confluence_pages.csv` not found
**Solution:** Run Step 1 (Dataset_Generator.ipynb) first

### Issue: `trained_t5_model` directory not found
**Solution:** Run Step 2 (T5_QnA_FineTuning.ipynb) and wait for training to complete

### Issue: Image file not found
**Solution:** Ensure `bot.jpg` exists at the specified path or update the path in the chatbot code

### Issue: Module not found errors
**Solution:** Install all required packages:
```bash
pip install pandas faker numpy transformers datasets torch sentence-transformers scikit-learn pillow
```

### Issue: Slow model loading
**Solution:** This is normal for first-time loading. Subsequent runs will be faster with cached models.

## Configuration

### Modify Dataset Size
In `Dataset_Generator.ipynb`, change the range value:
```python
for _ in range(1000):  # Change 1000 to desired number
```

### Adjust Training Parameters
In `T5_QnA_FineTuning.ipynb`, modify `TrainingArguments`:
```python
num_train_epochs=3,              # Increase for more training
per_device_train_batch_size=8,   # Adjust based on GPU memory
learning_rate=2e-5,              # Modify learning rate
```

### Change Image Path
In `Confluence_ChatBot.ipynb`, update:
```python
image_path = r'C:\Users\locperera\OneDrive\Documents\AI Hackathon\bot.jpg'
```

## License

Internal use only.
