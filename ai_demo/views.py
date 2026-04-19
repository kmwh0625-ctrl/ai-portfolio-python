import json
import re
import random
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
        decode_fn = tf.keras.applications.mobilenet_v2.decode_predictions
        img_bytes = image_file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32)[np.newaxis, ...]
        arr = preprocess(arr)
        preds = model.predict(arr)
        decoded = decode_fn(preds, top=5)[0]
        results = [{'label': label.replace('_', ' ').title(), 'score': round(float(score) * 100, 2)} for (_, label, score) in decoded]
        return JsonResponse({'success': True, 'results': results})
    except Exception:
        pass
    demo_results = [
        {'label': 'Golden Retriever', 'score': 82.4},
        {'label': 'Labrador Retriever', 'score': 9.1},
        {'label': 'Irish Setter', 'score': 3.2},
        {'label': 'Clumber Spaniel', 'score': 2.0},
        {'label': 'English Setter', 'score': 1.5},
    ]
    return JsonResponse({'success': True, 'results': demo_results, 'demo': True})


# ── 감성 분석 ─────────────────────────────────────────────────────────────────

def sentiment_analysis(request):
    return render(request, 'ai_demo/sentiment.html')

@csrf_exempt
def sentiment_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST 요청만 허용됩니다.'}, status=405)
    text = ''
    try:
        data = json.loads(request.body)
        text = str(data.get('text', '')).strip()
    except Exception:
        text = request.POST.get('text', '').strip()
    if not text:
        return JsonResponse({'error': '텍스트를 입력해주세요.'}, status=400)
    if len(text) > 1000:
        return JsonResponse({'error': '1000자 이하로 입력해주세요.'}, status=400)
    try:
        from transformers import pipeline as hf_pipeline
        clf = hf_pipeline('text-classification', model='snunlp/KR-FinBert-SC', return_all_scores=True, truncation=True, max_length=512)
        raw_results = clf(text[:512])[0]
        label_map = {'positive': '긍정 😊', 'negative': '부정 😔', 'neutral': '중립 😐'}
        formatted = sorted([{'label': label_map.get(r['label'].lower(), r['label']), 'score': round(r['score'] * 100, 2), 'raw': r['label'].lower()} for r in raw_results], key=lambda x: x['score'], reverse=True)
        return JsonResponse({'success': True, 'dominant': formatted[0], 'all_scores': formatted, 'text': text[:100] + ('...' if len(text) > 100 else '')})
    except Exception:
        pass
    pos_words = ['좋아','훌륭','최고','행복','기쁘','감사','사랑','완벽','좋은','즐거','설레','뿌듯','희망','기대','만족','성공','good','great','happy','love','excellent']
    neg_words = ['나쁘','싫어','최악','슬프','힘들','지쳤','화나','실망','끔찍','불만','잘못','우울','괴로','고통','절망','후회','bad','terrible','hate','awful','worst']
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
    formatted = [{'label': label_map[l], 'score': round(s * 100, 2), 'raw': l} for l, s in scores]
    return JsonResponse({'success': True, 'dominant': formatted[0], 'all_scores': formatted, 'text': text[:100] + ('...' if len(text) > 100 else ''), 'demo': True})


# ── 텍스트 요약 ───────────────────────────────────────────────────────────────

def text_summarize(request):
    return render(request, 'ai_demo/summarize.html')

@csrf_exempt
def summarize_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST 요청만 허용됩니다.'}, status=405)
    text = ''
    try:
        data = json.loads(request.body)
        text = str(data.get('text', '')).strip()
    except Exception:
        text = request.POST.get('text', '').strip()
    if not text:
        return JsonResponse({'error': '텍스트를 입력해주세요.'}, status=400)
    if len(text) < 100:
        return JsonResponse({'error': '100자 이상 입력해주세요.'}, status=400)
    if len(text) > 3000:
        return JsonResponse({'error': '3000자 이하로 입력해주세요.'}, status=400)
    try:
        from transformers import pipeline as hf_pipeline
        summarizer = hf_pipeline('summarization', model='gogamza/kobart-summarization', max_length=150, min_length=30, truncation=True)
        result = summarizer(text[:1024])[0]
        summary = result['summary_text']
        ratio = max(0, round((1 - len(summary) / len(text)) * 100))
        return JsonResponse({'success': True, 'summary': summary, 'original_len': len(text), 'summary_len': len(summary), 'ratio': ratio})
    except Exception:
        pass
    # 폴백: 문장 추출 요약
    sentences = [s.strip() for s in re.split(r'(?<=[.!?。])\s+', text.strip()) if len(s.strip()) > 10]
    if len(sentences) <= 3:
        summary = ' '.join(sentences)
    else:
        n = len(sentences)
        summary = ' '.join([sentences[0], sentences[n // 2], sentences[-1]])
    ratio = max(0, min(round((1 - len(summary) / len(text)) * 100), 90))
    return JsonResponse({'success': True, 'summary': summary, 'original_len': len(text), 'summary_len': len(summary), 'ratio': ratio, 'demo': True})


# ── 챗봇 ─────────────────────────────────────────────────────────────────────

def chatbot(request):
    return render(request, 'ai_demo/chatbot.html')

@csrf_exempt
def chatbot_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST 요청만 허용됩니다.'}, status=405)
    message = ''
    try:
        data = json.loads(request.body)
        message = str(data.get('message', '')).strip()
    except Exception:
        message = request.POST.get('message', '').strip()
    if not message:
        return JsonResponse({'error': '메시지를 입력해주세요.'}, status=400)
    if len(message) > 500:
        return JsonResponse({'error': '500자 이하로 입력해주세요.'}, status=400)

    msg_lower = message.lower()
    responses = {
        ('이력서', '자기소개서', '자소서'): "이력서 작성 팁을 알려드릴게요! 📝\n\n① 프로젝트 중심으로 쓰세요. 'Python 개발'보다 'Python으로 주가 분석 시스템 구현'처럼 구체적으로!\n② 수치를 넣으세요. '성능 30% 개선', '1만 건 데이터 처리' 등\n③ GitHub 링크를 꼭 넣으세요. 코드가 증거입니다.\n④ 1~2페이지로 간결하게 유지하세요.",
        ('면접', '준비', '질문'): "면접 준비 핵심을 알려드릴게요! 💼\n\n자주 나오는 질문:\n• '자기소개 해주세요' → 30초~1분, 지원 동기 포함\n• '프로젝트 설명해주세요' → 기술 스택 + 어려웠던 점 + 해결 과정\n• '부족한 점은?' → 솔직하게 + 개선 노력 함께\n\n기술 개념보다 왜 그 기술을 선택했는지가 더 중요해요!",
        ('django', '장고', 'python', '파이썬'): "Django/Python 관련 질문이군요! 🐍\n\nDjango 핵심 개념:\n• MTV 패턴 (Model-Template-View)\n• ORM으로 DB를 Python 코드로 조작\n• 관리자 페이지 자동 생성\n• 보안 기능 내장 (CSRF, XSS 방어)\n\n면접 답변: 'Django ORM이 뭔가요?' → 'SQL 없이 Python으로 DB를 다루는 기술'",
        ('ai', 'ml', '머신러닝', '딥러닝', '인공지능', 'tensorflow', '텐서플로'): "AI/ML 관련 내용이군요! 🤖\n\n신입 개발자 추천 AI 스택:\n• TensorFlow/PyTorch: 딥러닝 모델 구현\n• HuggingFace: 사전학습 모델 활용\n• scikit-learn: 머신러닝 기초\n• Pandas/NumPy: 데이터 전처리\n\nAI 프로젝트를 배포까지 해두면 큰 차별화 포인트가 돼요!",
        ('취업', '신입', '포트폴리오', '개발자'): "신입 개발자 취업 조언이에요! 🚀\n\n합격률을 높이는 3가지:\n① GitHub 꾸준히 커밋 → 잔디 심기\n② 배포된 프로젝트 → URL 공유 가능한 서비스\n③ 기술 블로그 → 배운 것 정리\n\n기술 스택보다 문제 해결 능력을 보여주는 게 핵심이에요!",
        ('안녕', '하이', 'hello', '반가'): "안녕하세요! 👋 저는 김원형 포트폴리오의 AI 챗봇이에요.\n\n취업 준비나 개발 관련 질문을 해보세요!\n\n예시:\n• '이력서 잘 쓰는 법 알려줘'\n• 'Django가 뭔가요?'\n• '신입 개발자 포트폴리오 팁'\n• '면접 준비 어떻게 해?'",
        ('크롤링', '스크래핑', 'beautifulsoup', 'selenium'): "웹 크롤링 관련이군요! 🕷️\n\n주요 도구:\n• requests + BeautifulSoup: 정적 웹사이트\n• Selenium: JS 렌더링 필요한 사이트\n• Scrapy: 대규모 크롤링\n\n주의사항:\n• robots.txt 확인 필수\n• time.sleep으로 서버 부하 방지\n\n크롤링 프로젝트는 자동화 능력을 잘 보여줄 수 있어요!",
        ('데이터', '분석', 'pandas', '시각화', 'matplotlib'): "데이터 분석 관련이군요! 📊\n\n기본 스택:\n• Pandas: 데이터 정제/분석\n• Matplotlib/Seaborn: 시각화\n• Plotly: 인터랙티브 차트\n• Jupyter Notebook: 분석 환경\n\n포트폴리오 팁: Jupyter Notebook으로 분석 과정을 보여주면 사고 과정이 드러나서 좋아요!",
    }

    reply = None
    for keywords, resp in responses.items():
        if any(kw in msg_lower for kw in keywords):
            reply = resp
            break

    if not reply:
        reply = f"'{message}'에 대해 답변드릴게요! 😊\n\n더 구체적으로 질문해주시면 더 잘 도와드릴 수 있어요.\n\n예시 질문:\n• '이력서 팁 알려줘'\n• '면접 준비 방법'\n• 'Django ORM이 뭔가요?'\n• '신입 포트폴리오 어떻게 만들어?'"

    return JsonResponse({'success': True, 'reply': reply, 'message': message})


# ── 주식 차트 데모 ────────────────────────────────────────────────────────────

def stock_demo(request):
    return render(request, 'ai_demo/stock.html')


@csrf_exempt
def stock_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        data = json.loads(request.body)
        ticker = data.get('ticker', '005930').strip().upper()
        period = data.get('period', '1mo')
    except Exception:
        ticker = '005930'
        period = '1mo'

    # 한국 주식 티커 매핑
    kr_map = {
        '삼성전자': '005930.KS', '카카오': '035720.KS', '네이버': '035420.KS',
        'SK하이닉스': '000660.KS', 'LG에너지솔루션': '373220.KS', '현대차': '005380.KS',
        'SAMSUNG': '005930.KS', 'KAKAO': '035720.KS', 'NAVER': '035420.KS',
    }
    if ticker in kr_map:
        ticker = kr_map[ticker]
    elif ticker.isdigit() and len(ticker) == 6:
        ticker = ticker + '.KS'

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            raise ValueError("데이터 없음")

        info = stock.info
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]
        closes = [round(float(p), 2) for p in hist['Close']]
        volumes = [int(v) for v in hist['Volume']]

        # 이동평균 계산
        ma5, ma20 = [], []
        for i in range(len(closes)):
            ma5.append(round(sum(closes[max(0,i-4):i+1]) / min(5, i+1), 2))
            ma20.append(round(sum(closes[max(0,i-19):i+1]) / min(20, i+1), 2))

        change = closes[-1] - closes[-2] if len(closes) >= 2 else 0
        change_pct = round(change / closes[-2] * 100, 2) if len(closes) >= 2 else 0

        return JsonResponse({
            'success': True,
            'ticker': ticker,
            'name': info.get('longName', info.get('shortName', ticker)),
            'current': closes[-1],
            'change': round(change, 2),
            'change_pct': change_pct,
            'high52': info.get('fiftyTwoWeekHigh'),
            'low52': info.get('fiftyTwoWeekLow'),
            'dates': dates,
            'closes': closes,
            'ma5': ma5,
            'ma20': ma20,
            'volumes': volumes,
        })

    except ImportError:
        # yfinance 없을 때 데모 데이터
        import random
        base = 75000
        dates, closes, volumes = [], [], []
        days = 30 if period == '1mo' else (90 if period == '3mo' else 365)
        for i in range(days):
            from datetime import date, timedelta
            d = date.today() - timedelta(days=days-i)
            dates.append(d.strftime('%Y-%m-%d'))
            base = base + random.randint(-2000, 2200)
            closes.append(max(60000, base))
            volumes.append(random.randint(8000000, 20000000))

        ma5 = [round(sum(closes[max(0,i-4):i+1])/min(5,i+1),2) for i in range(len(closes))]
        ma20 = [round(sum(closes[max(0,i-19):i+1])/min(20,i+1),2) for i in range(len(closes))]
        change = closes[-1] - closes[-2]

        return JsonResponse({
            'success': True, 'demo': True,
            'ticker': '005930.KS', 'name': '삼성전자 (데모)',
            'current': closes[-1], 'change': round(change,2),
            'change_pct': round(change/closes[-2]*100,2),
            'high52': max(closes), 'low52': min(closes),
            'dates': dates, 'closes': closes,
            'ma5': ma5, 'ma20': ma20, 'volumes': volumes,
        })

    except Exception as e:
        return JsonResponse({'error': f'데이터 조회 실패: {str(e)}'}, status=500)


# ── 뉴스 크롤링 데모 ──────────────────────────────────────────────────────────

def news_demo(request):
    return render(request, 'ai_demo/news.html')


@csrf_exempt
def news_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        data = json.loads(request.body)
        keyword = data.get('keyword', 'AI').strip()
        if not keyword:
            keyword = 'AI'
    except Exception:
        keyword = 'AI'

    if len(keyword) > 50:
        return JsonResponse({'error': '키워드는 50자 이하로 입력해주세요.'}, status=400)

    try:
        import requests as req
        from bs4 import BeautifulSoup

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        url = f'https://search.naver.com/search.naver?where=news&query={keyword}&sort=1&ds=&de=&nso=so%3Add%2Cp%3Aall'
        resp = req.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(resp.text, 'html.parser')

        articles = []
        items = soup.select('.news_wrap')[:8]

        for item in items:
            title_el = item.select_one('.news_tit')
            desc_el = item.select_one('.dsc_wrap') or item.select_one('.api_txt_lines')
            info_el = item.select_one('.info_group') or item.select_one('.news_info')
            press_el = item.select_one('.info_press') or item.select_one('.press')
            date_el = item.select_one('.info_group span:last-child') or item.select_one('span.info')

            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            link = title_el.get('href', '#')
            desc = desc_el.get_text(strip=True)[:120] + '...' if desc_el else '내용 없음'
            press = press_el.get_text(strip=True) if press_el else '언론사 미상'
            date = date_el.get_text(strip=True) if date_el else ''

            articles.append({
                'title': title,
                'link': link,
                'desc': desc,
                'press': press,
                'date': date,
            })

        if not articles:
            raise ValueError("기사를 찾을 수 없습니다")

        return JsonResponse({
            'success': True,
            'keyword': keyword,
            'count': len(articles),
            'articles': articles,
            'source': '네이버 뉴스',
        })

    except ImportError:
        pass
    except Exception:
        pass

    # 폴백: 더미 뉴스 데이터
    dummy = [
        {'title': f'[{keyword}] 최신 기술 동향과 미래 전망', 'link': '#', 'press': '연합뉴스', 'date': '1시간 전', 'desc': f'{keyword} 분야에서 새로운 기술적 진보가 이루어지고 있습니다. 전문가들은 이번 발전이 산업 전반에 큰 영향을 미칠 것으로 전망합니다.'},
        {'title': f'{keyword} 시장 급성장... 기업들 앞다퉈 투자', 'link': '#', 'press': '조선일보', 'date': '2시간 전', 'desc': f'국내외 주요 기업들이 {keyword} 기술에 대한 투자를 확대하고 있습니다. 시장 규모가 올해 대비 2배 이상 성장할 것으로 예측됩니다.'},
        {'title': f'정부, {keyword} 육성 정책 발표... 1조원 투자 계획', 'link': '#', 'press': '한겨레', 'date': '3시간 전', 'desc': f'정부가 {keyword} 분야 육성을 위한 대규모 투자 계획을 발표했습니다. 스타트업 지원과 인재 양성에 초점을 맞출 예정입니다.'},
        {'title': f'{keyword} 스타트업 시리즈A 투자 유치 성공', 'link': '#', 'press': '매일경제', 'date': '5시간 전', 'desc': f'{keyword} 전문 스타트업이 100억원 규모의 시리즈A 투자를 유치했습니다. 글로벌 시장 진출을 목표로 하고 있습니다.'},
        {'title': f'대학교 {keyword} 관련 학과 신설 붐... 취업률 90% 육박', 'link': '#', 'press': '동아일보', 'date': '7시간 전', 'desc': f'{keyword} 관련 학과를 신설하는 대학이 늘어나고 있습니다. 높은 취업률로 인해 입학 경쟁도 치열해지고 있습니다.'},
    ]

    return JsonResponse({
        'success': True, 'demo': True,
        'keyword': keyword,
        'count': len(dummy),
        'articles': dummy,
        'source': '네이버 뉴스 (데모)',
    })
