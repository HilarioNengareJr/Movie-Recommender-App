#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run the application
gunicorn wsgi:app
