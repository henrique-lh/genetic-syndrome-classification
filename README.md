# Genetic Syndrome Classification

**Context**
You are being hired by a fictional biotech company specializing in genetic research. The task involves analyzing embeddings derived from images to classify genetic syndromes. These embeddings are outputs from a pre-trained classification model. The company wants to improve its understanding of the data distribution and enhance the classification accuracy of genetic syndromes based on these
embeddings.
Your objective is to implement a comprehensive pipeline that includes data preprocessing, visualization, classification, manual implementation of key metrics, and insightful analysis.

---

## Prerequisites

You must have installed:
- Python 3.11.10

Optional:
- UV

## Setup

In the root of the project you must run:
```bash
# If you have python UV installed
uv sync

# If you have only pyhton and pip installed
pip install -r requirements.txt
```

## Usage

### Starting the Server

From the project root directory:

```bash
# Run pipeline and then start server (default)
python main.py

# Run only the inference server
python main.py --server

# Start server on a specific port
python main.py --server --port 9000

# Run only the pipeline
python main.py --pipeline

# Run pipeline and then start server
python main.py --all
```

### Testing with cURL

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Single prediction with sample data
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @artifacts/sample_inputs/sample_1.json

# Batch prediction
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d @artifacts/sample_inputs/batch_input.json

# Get model info
curl -X GET "http://localhost:8000/model/info"
```

## Troubleshooting

### "Model not loaded" Error
**Cause:** The trained model file is missing.  
**Solution:** Run the pipeline first:
```bash
python main.py --pipeline
```

### Port Already in Use
**Cause:** Another process is using port 8000.  
**Solution:** Use a different port:
```bash
python main.py --server --port 9000
```

## Reports

Check the REPORTS.md file with all reports about the problem, solutions, analysis, and possible improvements