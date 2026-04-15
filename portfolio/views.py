from django.shortcuts import render
from .models import Project, Skill


def index(request):
    featured_projects = Project.objects.filter(is_featured=True)[:3]
    all_projects = Project.objects.all()
    skills = Skill.objects.all()

    skill_categories = {}
    for skill in skills:
        if skill.category not in skill_categories:
            skill_categories[skill.category] = []
        skill_categories[skill.category].append(skill)

    context = {
        'featured_projects': featured_projects,
        'all_projects': all_projects,
        'skill_categories': skill_categories,
    }
    return render(request, 'portfolio/index.html', context)


def project_detail(request, pk):
    from django.shortcuts import get_object_or_404
    project = get_object_or_404(Project, pk=pk)
    related = Project.objects.filter(category=project.category).exclude(pk=pk)[:3]
    return render(request, 'portfolio/project_detail.html', {'project': project, 'related': related})