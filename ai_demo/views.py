import json
import base64
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# ── Image Classification ──────────────────────────────────────────────────────

def image_classify(request):
    return render(request, 'ai_demo/image_classify.html')


@csrf_exempt
def classify_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST 요청만 허용됩니다.'}, status=405)

    image_file = request.FILES.get('image')
    if not image_file:
        return JsonResponse({'error': '이미지를 업로드해주세요.'}, status=400)

    # Check size (max 5MB)
    if image_file.size > 5 * 1024 * 1024:
        return JsonResponse({'error': '파일 크기는 5MB 이하여야 합니다.'}, status=400)

    try:
        import tensorflow as tf
        import numpy as np
        from PIL import Image
        import io

        # Load MobileNetV2
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        decode = tf.keras.applications.mobilenet_v2.decode_predictions

        # Preprocess image
        img_bytes = image_file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32)[np.newaxis, ...]
        arr = preprocess(arr)

        preds = model.predict(arr)
        decoded = decode(preds, top=5)[0]

        results = [
            {'label': label.replace('_', ' ').title(), 'score': round(float(score) * 100, 2)}
            for (_, label, score) in decoded
        ]
        return JsonResponse({'success': True, 'results': results})

    except ImportError:
        # TensorFlow not installed – return demo data
        results = [
            {'label': 'Golden Retriever', 'score': 82.4},
            {'label': 'Labrador Retriever', 'score': 9.1},
            {'label': 'Irish Setter', 'score': 3.2},
            {'label': 'Clumber Spaniel', 'score': 2.0},
            {'label': 'English Setter', 'score': 1.5},
        ]
        return JsonResponse({'success': True, 'results': results, 'demo': True})

    except Exception as e:
        return JsonResponse({'error': f'분류 중 오류가 발생했습니다: {str(e)}'}, status=500)


# ── Sentiment Analysis ────────────────────────────────────────────────────────

def sentiment_analysis(request):
    return render(request, 'ai_demo/sentiment.html')


@csrf_exempt
def sentiment_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST 요청만 허용됩니다.'}, status=405)

    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
    except (json.JSONDecodeError, AttributeError):
        text = request.POST.get('text', '').strip()

    if not text:
        return JsonResponse({'error': '텍스트를 입력해주세요.'}, status=400)

    if len(text) > 1000:
        return JsonResponse({'error': '텍스트는 1000자 이하로 입력해주세요.'}, status=400)

    try:
        from transformers import pipeline

        classifier = pipeline(
            'sentiment-analysis',
            model='snunlp/KR-FinBert-SC',
            return_all_scores=True
        )
        results = classifier(text[:512])[0]

        label_map = {'positive': '긍정 😊', 'negative': '부정 😔', 'neutral': '중립 😐'}
        formatted = [
            {
                'label': label_map.get(r['label'].lower(), r['label']),
                'score': round(r['score'] * 100, 2),
                'raw': r['label'].lower()
            }
            for r in sorted(results, key=lambda x: x['score'], reverse=True)
        ]
        dominant = formatted[0]
        return JsonResponse({
            'success': True,
            'dominant': dominant,
            'all_scores': formatted,
            'text': text[:100] + ('...' if len(text) > 100 else ''),
        })

    except ImportError:
        # HuggingFace not installed – keyword-based demo
        pos_words = ['좋아', '훌륭', '최고', '행복', '기쁘', '감사', '사랑', '완벽', '좋은', 'good', 'great', 'happy', 'love', 'excellent']
        neg_words = ['나쁘', '싫어', '최악', '슬프', '화나', '실망', '끔찍', '불만', 'bad', 'terrible', 'hate', 'awful', 'worst']

        text_lower = text.lower()
        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)

        if pos_count > neg_count:
            dominant_raw = 'positive'
            scores = [('positive', 0.78), ('neutral', 0.15), ('negative', 0.07)]
        elif neg_count > pos_count:
            dominant_raw = 'negative'
            scores = [('negative', 0.75), ('neutral', 0.17), ('positive', 0.08)]
        else:
            dominant_raw = 'neutral'
            scores = [('neutral', 0.65), ('positive', 0.20), ('negative', 0.15)]

        label_map = {'positive': '긍정 😊', 'negative': '부정 😔', 'neutral': '중립 😐'}
        formatted = [{'label': label_map[l], 'score': round(s * 100, 2), 'raw': l} for l, s in scores]

        return JsonResponse({
            'success': True,
            'dominant': formatted[0],
            'all_scores': formatted,
            'text': text[:100] + ('...' if len(text) > 100 else ''),
            'demo': True,
        })

    except Exception as e:
        return JsonResponse({'error': f'분석 중 오류: {str(e)}'}, status=500)