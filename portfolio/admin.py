from django.contrib import admin
from .models import Project, Skill


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ['title', 'category', 'is_featured', 'order', 'created_at']
    list_editable = ['is_featured', 'order']
    list_filter = ['category', 'is_featured']
    search_fields = ['title', 'description']


@admin.register(Skill)
class SkillAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'proficiency', 'order']
    list_editable = ['proficiency', 'order']
    list_filter = ['category']