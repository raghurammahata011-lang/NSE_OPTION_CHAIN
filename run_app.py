import subprocess
import sys
import os
import webbrowser
import time

# Install TA-Lib wheel if not already installed
wheel_path = os.path.join(os.path.dirname(__file__), "TA_Lib-0.6.4-cp312-cp312-win_amd64.whl")
try:
    import talib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path])

# Path to main Streamlit app
app_path = os.path.join(os.path.dirname(__file__), "NSE_MKT_ANALYSIS_V2.py")

# Launch Streamlit in headless mode
subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", app_path,
    "--server.headless", "true", "--server.port", "8501"
])

# Wait a few seconds for Streamlit to start
time.sleep(5)

# Open the browser automatically
webbrowser.open("http://localhost:8501")
