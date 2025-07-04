import os
import logging
from flask import Flask, request, jsonify
import requests
from inference_functions import (
  convert_predictions_to_aspects,
  aspects_extraction,
  classify_product,
  classify_sentiment
)
from transformers import (
  BertTokenizerFast,
  BertForTokenClassification,
  BertTokenizer,
  BertForSequenceClassification,
  AutoModelForSequenceClassification,
  AutoTokenizer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize global variables
tokenizer_sent = None
tokenizer_asp = None  
tokenizer_cls = None
AspectModel = None
SentimentModel = None
ClassificationModel = None

def load_models():
    """Load all models and tokenizers once at startup"""
    global tokenizer_sent, tokenizer_asp, tokenizer_cls
    global AspectModel, SentimentModel, ClassificationModel
    
    try:
        logger.info("Starting model loading...")
        
        # Check if model directories exist
        import os
        model_paths = [
            "absa_model/checkpoint-10456",
            "aspect_extraction_model/checkpoint-16276", 
            "product_classifier"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                logger.info(f"✓ Found model directory: {path}")
            else:
                logger.error(f"✗ Missing model directory: {path}")
        
        # Load tokenizers
        logger.info("Loading tokenizers...")
        tokenizer_sent = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer_asp = BertTokenizerFast.from_pretrained("bert-base-uncased")
        tokenizer_cls = AutoTokenizer.from_pretrained("roberta-base")
        logger.info("✓ Tokenizers loaded")
        
        # Load models
        logger.info("Loading aspect model...")
        AspectModel = BertForTokenClassification.from_pretrained("absa_model/checkpoint-10456")
        logger.info("✓ Aspect model loaded")
        
        logger.info("Loading sentiment model...")
        SentimentModel = BertForSequenceClassification.from_pretrained("aspect_extraction_model/checkpoint-16276")
        logger.info("✓ Sentiment model loaded")
        
        logger.info("Loading classification model...")
        ClassificationModel = AutoModelForSequenceClassification.from_pretrained("product_classifier")
        logger.info("✓ Classification model loaded")

        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Directory contents: {os.listdir('.')}")
        # Don't raise - let the app start but mark models as not loaded
        return False
    
    return True

# Load models after defining the function but before routes
load_models()

@app.route('/')
def home():
    return jsonify({
        'message': 'AI Model API is running!',
        'endpoints': {
            'analyze': 'POST /analyze',
            'health': 'GET /health'
        }
    })

def process_request(request):
    try:
        # Extract aspects
        aspects = aspects_extraction(request[0][0]['Review'], AspectModel, tokenizer_asp)
        
        # Classify product
        product = classify_product(request[0]['Review'], ClassificationModel, tokenizer_cls)
        
        # Get sentiments for each aspect
        sentiments = []
        for aspect in aspects:
            sentiment = classify_sentiment(request[0]['Review'], aspect['term'], SentimentModel, tokenizer_sent)
            sentiments.append(sentiment)
        
        return {
            'feedback': request[0]['Review'],
            'aspects': aspects,
            'sentiments': sentiments,
            'product': product
        }

    except Exception as e:
        logger.error(f"Error processing review: {str(e)}")
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': all([
            tokenizer_sent is not None,
            tokenizer_asp is not None,
            tokenizer_cls is not None,
            AspectModel is not None,
            SentimentModel is not None,
            ClassificationModel is not None,
        ]),
        'individual_models': {
            'tokenizer_sent': tokenizer_sent is not None,
            'tokenizer_asp': tokenizer_asp is not None,
            'tokenizer_cls': tokenizer_cls is not None,
            'AspectModel': AspectModel is not None,
            'SentimentModel': SentimentModel is not None,
            'ClassificationModel': ClassificationModel is not None,
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Main endpoint for text analysis"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'Review' not in data:
            return jsonify({'error': 'Missing "Review" field in request'}), 400
        
        review = data['Review']
        
        if not review or not isinstance(review, str):
            return jsonify({'error': 'Invalid review provided'}), 400
        
        # Process the review
        result = process_request(review)

        # Post each aspect (topic) individually
        target_url = "https://ba7c-37-114-161-59.ngrok-free.app/api/v0.1/reviews"
        responses = []
        import datetime
        aspects = result.get('aspects', [])
        sentiments = result.get('sentiments', [])
        for idx, aspect in enumerate(aspects):
            # Map sentiment string to required integer
            sentiment_str = sentiments[idx] if idx < len(sentiments) else ""
            if sentiment_str == "positive":
                sentiment_val = 1
            elif sentiment_str == "negative":
                sentiment_val = 2
            elif sentiment_str == "neutral":
                sentiment_val = 3
            else:
                sentiment_val = 0  # fallback for unknown

            payload = {
                "customerId": 0,
                "productId": 0,
                "topic": aspect.get('term', ""),
                "sentiment": sentiment_val,
            }
            try:
                logger.info(f"Sending payload to {target_url}: {payload}")
                response = requests.post(
                    target_url,
                    json=payload,
                    headers={
                        "accept": "text/plain",
                        "Content-Type": "application/json"
                    }
                )
                logger.info(f"Response status: {response.status_code}, content: {response.text}")
                response.raise_for_status()
                responses.append({"topic": payload["topic"], "status": "success", "response": response.text})
            except Exception as post_err:
                logger.error(f"Error posting to client for topic '{payload['topic']}': {str(post_err)}")
                responses.append({"topic": payload["topic"], "status": "failed", "error": str(post_err)})

        return jsonify({'success': True, 'results': responses})
    except Exception as e:
        logger.error(f"Error in analyze_text: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/test', methods=['POST'])
def test_endpoint():
    """Test endpoint to verify functionality"""
    test_review = "I don't feel that touch screen works perfectly"
    try:
        result = process_request([{'Review': test_review}])
        target_url = "https://assured-goshawk-topical.ngrok-free.app/api/v0.1/reviews"
        aspects = result.get('aspects', [])
        sentiments = result.get('sentiments', [])
        responses = []
        
        for idx, aspect in enumerate(aspects):
            sentiment_str = sentiments[idx] if idx < len(sentiments) else ""
            if sentiment_str == "positive":
                sentiment_val = 1
            elif sentiment_str == "negative":
                sentiment_val = 2
            elif sentiment_str == "neutral":
                sentiment_val = 3
            else:
                sentiment_val = 0
                
            payload = {
                "customerId": 1,
                "productId": 1,
                "topic": aspect.get('term', ""),
                "sentiment": sentiment_val,
            }
            
            logger.info(f"[TEST] Sending payload to {target_url}: {payload}")
            response = requests.post(
                target_url,
                json=payload,
                headers={
                    "accept": "text/plain",
                    "Content-Type": "application/json"
                }
            )
            logger.info(f"[TEST] Response status: {response.status_code}, content: {response.text}")
            responses.append({
                "payload": payload,
                "status": response.status_code,
                "response": response.text
            })
            
        return jsonify({
            "test_review": test_review,
            "result": result,
            "responses": responses
        })
        
    except Exception as e:
        logger.error(f"[TEST] Error in test endpoint: {str(e)}")

        return jsonify({"error": str(e)}), 500
      
if __name__ == '__main__':
  port = int(os.environ.get('PORT', 10000))  # Hugging Face uses port 7860
  app.run(host='0.0.0.0', port=port)
