import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


# ─────────────────────────────────────────────────────────────────────────────
# 이미지 분류
# ─────────────────────────────────────────────────────────────────────────────

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

    # TensorFlow 시도
    try:
        import tensorflow as tf
        import numpy as np
        from PIL import Image
        import io

        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        decode_fn = tf.keras.applications.mobilenet_v2.decode_predictions

        img_bytes = image_file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32)[np.newaxis, ...]
        arr = preprocess(arr)
        preds = model.predict(arr)
        decoded = decode_fn(preds, top=5)[0]

        results = [
            {'label': label.replace('_', ' ').title(), 'score': round(float(score) * 100, 2)}
            for (_, label, score) in decoded
        ]
        return JsonResponse({'success': True, 'results': results})

    except ImportError:
        pass  # TF 없으면 데모 데이터
    except Exception:
        pass  # 그 외 오류도 데모 데이터로 폴백

    # 데모 데이터 (TF 미설치 환경)
    demo_results = [
        {'label': 'Golden Retriever', 'score': 82.4},
        {'label': 'Labrador Retriever', 'score': 9.1},
        {'label': 'Irish Setter', 'score': 3.2},
        {'label': 'Clumber Spaniel', 'score': 2.0},
        {'label': 'English Setter', 'score': 1.5},
    ]
    return JsonResponse({'success': True, 'results': demo_results, 'demo': True})


# ─────────────────────────────────────────────────────────────────────────────
# 감성 분석  ← 이게 핵심 수정 부분
# ─────────────────────────────────────────────────────────────────────────────

def sentiment_analysis(request):
    return render(request, 'ai_demo/sentiment.html')


@csrf_exempt
def sentiment_api(request):
    """
    절대 500 에러를 내지 않는 감성 분석 API.
    HuggingFace → 키워드 폴백 → 기본값 순서로 항상 success: true 반환.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST 요청만 허용됩니다.'}, status=405)

    # ── 텍스트 파싱 (JSON / form 둘 다 허용) ──────────────────────────────────
    text = ''
    try:
        body = request.body
        if body:
            data = json.loads(body)
            text = str(data.get('text', '')).strip()
    except Exception:
        pass

    if not text:
        text = request.POST.get('text', '').strip()

    if not text:
        return JsonResponse({'error': '텍스트를 입력해주세요.'}, status=400)

    if len(text) > 1000:
        return JsonResponse({'error': '1000자 이하로 입력해주세요.'}, status=400)

    # ── 1순위: HuggingFace transformers ───────────────────────────────────────
    try:
        from transformers import pipeline as hf_pipeline

        clf = hf_pipeline(
            'text-classification',
            model='snunlp/KR-FinBert-SC',
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )
        raw_results = clf(text[:512])[0]

        label_map = {
            'positive': '긍정 😊',
            'negative': '부정 😔',
            'neutral':  '중립 😐',
        }
        formatted = sorted(
            [
                {
                    'label': label_map.get(r['label'].lower(), r['label']),
                    'score': round(r['score'] * 100, 2),
                    'raw':   r['label'].lower(),
                }
                for r in raw_results
            ],
            key=lambda x: x['score'],
            reverse=True,
        )
        return JsonResponse({
            'success':    True,
            'dominant':   formatted[0],
            'all_scores': formatted,
            'text':       text[:100] + ('...' if len(text) > 100 else ''),
        })

    except Exception:
        pass  # 어떤 오류든 키워드 폴백으로

    # ── 2순위: 키워드 기반 폴백 (항상 성공) ──────────────────────────────────
    pos_words = [
        '좋아', '훌륭', '최고', '행복', '기쁘', '감사', '사랑', '완벽', '좋은',
        '즐거', '설레', '뿌듯', '희망', '기대', '만족', '성공',
        'good', 'great', 'happy', 'love', 'excellent', 'wonderful', 'amazing',
    ]
    neg_words = [
        '나쁘', '싫어', '최악', '슬프', '힘들', '지쳤', '화나', '실망', '끔찍',
        '불만', '잘못', '우울', '괴로', '힘내', '고통', '절망', '후회',
        'bad', 'terrible', 'hate', 'awful', 'worst', 'horrible', 'sad',
    ]

    tl = text.lower()
    pos_count = sum(1 for w in pos_words if w in tl)
    neg_count = sum(1 for w in neg_words if w in tl)

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
        'success':    True,
        'dominant':   formatted[0],
        'all_scores': formatted,
        'text':       text[:100] + ('...' if len(text) > 100 else ''),
        'demo':       True,
    })