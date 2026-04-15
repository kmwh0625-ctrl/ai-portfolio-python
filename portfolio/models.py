from django.db import models


class Project(models.Model):
    CATEGORY_CHOICES = [
        ('ai', 'AI / Machine Learning'),
        ('web', '웹 개발'),
        ('data', '데이터 분석'),
        ('other', '기타'),
    ]

    title = models.CharField(max_length=200, verbose_name='프로젝트명')
    subtitle = models.CharField(max_length=300, blank=True, verbose_name='부제목')
    description = models.TextField(verbose_name='설명')
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default='ai', verbose_name='카테고리')
    tech_stack = models.CharField(max_length=500, verbose_name='기술 스택', help_text='쉼표로 구분하여 입력')
    github_url = models.URLField(blank=True, verbose_name='GitHub URL')
    demo_url = models.URLField(blank=True, verbose_name='데모 URL')
    image = models.ImageField(upload_to='projects/', blank=True, null=True, verbose_name='프로젝트 이미지')
    is_featured = models.BooleanField(default=False, verbose_name='주요 프로젝트')
    order = models.IntegerField(default=0, verbose_name='순서')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['order', '-created_at']
        verbose_name = '프로젝트'
        verbose_name_plural = '프로젝트 목록'

    def __str__(self):
        return self.title

    def get_tech_list(self):
        return [t.strip() for t in self.tech_stack.split(',') if t.strip()]


class Skill(models.Model):
    name = models.CharField(max_length=100, verbose_name='기술명')
    proficiency = models.IntegerField(default=80, verbose_name='숙련도 (%)')
    category = models.CharField(max_length=100, verbose_name='카테고리')
    order = models.IntegerField(default=0)

    class Meta:
        ordering = ['category', 'order']
        verbose_name = '기술'
        verbose_name_plural = '기술 목록'

    def __str__(self):
        return self.name