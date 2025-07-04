# AI Text Analysis API

This is a Flask API that provides text analysis using BERT models for:
- Aspect extraction
- Sentiment analysis
- Product classification

## API Endpoints

### POST /analyze
Analyze text input.

**Request:**
```json
{
  "sentence": "I don't feel that touch screen works perfectly"
}