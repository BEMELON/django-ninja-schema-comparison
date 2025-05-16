"""
Minimal Django settings configuration for using django-ninja in a notebook environment.
"""
import os
from pathlib import Path

# Build paths inside the project
BASE_DIR = Path(__file__).resolve().parent

# Quick-start development settings - unsuitable for production
SECRET_KEY = 'django-insecure-dummy-key-for-notebook-use-only'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.auth',
]

# Django Ninja settings
NINJA_PAGINATION_CLASS = 'ninja.pagination.PageNumberPagination'
NINJA_PAGINATION_PER_PAGE = 100
NINJA_MAX_PER_PAGE_SIZE = 1000
NINJA_PAGINATION_MAX_LIMIT = 1000
NINJA_NUM_PROXIES = None
NINJA_DEFAULT_THROTTLE_RATES = {}
NINJA_FIX_REQUEST_FILES_METHODS = set()

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
