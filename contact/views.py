from django.shortcuts import render, redirect
from django.contrib import messages
from .models import ContactMessage

def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        email = request.POST.get('email', '').strip()
        subject = request.POST.get('subject', '').strip()
        message = request.POST.get('message', '').strip()

        errors = {}
        if not name:
            errors['name'] = '이름을 입력해주세요.'
        if not email:
            errors['email'] = '이메일을 입력해주세요.'
        if not subject:
            errors['subject'] = '제목을 입력해주세요.'
        if not message:
            errors['message'] = '메시지를 입력해주세요.'

        if not errors:
            ContactMessage.objects.create(
                name=name,
                email=email,
                subject=subject,
                message=message,
            )
            messages.success(request, '메시지가 성공적으로 전송되었습니다. 빠른 시일 내에 답변드리겠습니다!')
            return redirect('contact:contact')

        return render(request, 'contact/contact.html', {
            'errors': errors,
            'form_data': {'name': name, 'email': email, 'subject': subject, 'message': message}
        })

    return render(request, 'contact/contact.html')