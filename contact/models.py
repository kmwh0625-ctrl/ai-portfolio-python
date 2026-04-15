from django.db import models


class ContactMessage(models.Model):
    name = models.CharField(max_length=100, verbose_name='이름')
    email = models.EmailField(verbose_name='이메일')
    subject = models.CharField(max_length=200, verbose_name='제목')
    message = models.TextField(verbose_name='메시지')
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False, verbose_name='읽음')

    class Meta:
        ordering = ['-created_at']
        verbose_name = '문의 메시지'
        verbose_name_plural = '문의 메시지 목록'

    def __str__(self):
        return f'[{self.name}] {self.subject}'