import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


# ── 이미지 분류 ───────────────────────────────────────────────────────────────

def image_classify(request):
    return render(request, 'ai_demo/image_classify.html')


@csrf_exempt
def classify_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST 요청만 허용됩니다.'}, status=405)

    image_file = request.FILES.get('image')
    if not image_file:
        return JsonResponse({'error': '이미지를 업로드해주세요.'}, status=400)

    if image_file.size > 5 * 1024 * 1024:
        return JsonResponse({'error': '파일 크기는 5MB 이하여야 합니다.'}, status=400)

    try:
        import tensorflow as tf
        import numpy as np
        from PIL import Image
        import io

        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        decode = tf.keras.applications.mobilenet_v2.decode_predictions

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
        # TensorFlow 미설치 → 데모 데이터 반환
        results = [
            {'label': 'Golden Retriever', 'score': 82.4},
            {'label': 'Labrador Retriever', 'score': 9.1},
            {'label': 'Irish Setter', 'score': 3.2},
            {'label': 'Clumber Spaniel', 'score': 2.0},
            {'label': 'English Setter', 'score': 1.5},
        ]
        return JsonResponse({'success': True, 'results': results, 'demo': True})

    except Exception as e:
        return JsonResponse({'success': False, 'error': f'분류 중 오류: {str(e)}'}, status=500)


# ── 감성 분석 ─────────────────────────────────────────────────────────────────

def sentiment_analysis(request):
    return render(request, 'ai_demo/sentiment.html')


@csrf_exempt
def sentiment_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST 요청만 허용됩니다.'}, status=405)

    # body 파싱 - JSON이든 form이든 둘 다 처리
    text = ''
    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
    except Exception:
        text = request.POST.get('text', '').strip()

    if not text:
        return JsonResponse({'error': '텍스트를 입력해주세요.'}, status=400)

    if len(text) > 1000:
        return JsonResponse({'error': '텍스트는 1000자 이하로 입력해주세요.'}, status=400)

    # HuggingFace 시도
    try:
        from transformers import pipeline
        classifier = pipeline(
            'sentiment-analysis',
            model='snunlp/KR-FinBert-SC',
            return_all_scores=True
        )
        results = classifier(text[:512])[0]
        label_map = {'positive': '긍정 😊', 'negative': '부정 😔', 'neutral': '중립 😐'}
        formatted = sorted(
            [
                {
                    'label': label_map.get(r['label'].lower(), r['label']),
                    'score': round(r['score'] * 100, 2),
                    'raw': r['label'].lower()
                }
                for r in results
            ],
            key=lambda x: x['score'], reverse=True
        )
        return JsonResponse({
            'success': True,
            'dominant': formatted[0],
            'all_scores': formatted,
            'text': text[:100] + ('...' if len(text) > 100 else ''),
        })

    except ImportError:
        pass  # HuggingFace 없으면 아래 키워드 방식으로 대체
    except Exception as e:
        # 모델 로딩 오류 등 → 키워드 방식으로 대체
        pass

    # ── 키워드 기반 폴백 (항상 성공 보장) ──────────────────────────────────────
    pos_words = ['좋아', '훌륭', '최고', '행복', '기쁘', '감사', '사랑', '완벽', '좋은',
                 'good', 'great', 'happy', 'love', 'excellent', '기대', '설레', '뿌듯']
    neg_words = ['나쁘', '싫어', '최악', '슬프', '힘들', '지쳤', '화나', '실망', '끔찍',
                 '불만', '잘못', 'bad', 'terrible', 'hate', 'awful', 'worst', '우울', '괴로']

    text_lower = text.lower()
    pos_count = sum(1 for w in pos_words if w in text_lower)
    neg_count = sum(1 for w in neg_words if w in text_lower)

    if pos_count > neg_count:
        scores = [('positive', 0.78), ('neutral', 0.15), ('negative', 0.07)]
    elif neg_count > pos_count:
        scores = [('negative', 0.75), ('neutral', 0.17), ('positive', 0.08)]
    else:
        scores = [('neutral', 0.60), ('positive', 0.22), ('negative', 0.18)]

    label_map = {'positive': '긍정 😊', 'negative': '부정 😔', 'neutral': '중립 😐'}
    formatted = [
        {'label': label_map[l], 'score': round(s * 100, 2), 'raw': l}
        for l, s in scores
    ]

    return JsonResponse({
        'success': True,
        'dominant': formatted[0],
        'all_scores': formatted,
        'text': text[:100] + ('...' if len(text) > 100 else ''),
        'demo': True,
    })