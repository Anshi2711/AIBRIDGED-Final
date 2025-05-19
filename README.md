# AIBRIDGED-Final
# Step 1: Navigate to your project folder
cd C:\Apps\AIBRIDGED-Final\AI-News-Summariser-main

# Step 2: Delete old virtual environment if it exists (optional)
Remove-Item -Recurse -Force .\venv\

# Step 3: Create a new virtual environment
python -m venv venv

# Step 4: Activate the virtual environment
.\venv\Scripts\activate

# Step 5: Edit requirements.txt → REMOVE or COMMENT OUT the line: sentencepiece

# Step 6: Install all other dependencies
pip install -r requirements.txt

# Step 7: Install sentencepiece manually (this version works reliably)
pip install sentencepiece==0.1.99

# Step 8: Run the Flask app
python app.py
