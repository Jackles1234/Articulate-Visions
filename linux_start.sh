#!/bin/bash

# Check Requirements
command -v python3 >/dev/null 2>&1 && echo "Python 3 is installed"
command -v npm >/dev/null 2>&1 && echo "NPM installed"

# starts the backend
pip install -r backend/requirements.txt
python3 backend/app.py && echo "Backend running"

# runs in the frontend folder, just starts frontend and opens the browser
cd frontend || exit
npm install
xdg-open http://localhost:3000/

npm run build && npm run start



