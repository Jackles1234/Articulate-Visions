import subprocess
import os

try:
    subprocess.call(['pip', 'install', '-r', 'backend/requirements.txt'])
    from backend import app
    app.main()
except Exception as e:
    print(f"Could not start backend, exiting \n Error: {e}")

os.chdir('./frontend')

try:
    subprocess.call(['npm', 'install'])
    subprocess.call(['npm', 'run', 'build'])
    subprocess.call(['npm', 'run', 'start'])
except Exception as e:
    print(f"Could not start frontend, exiting \n Error: {e}")
