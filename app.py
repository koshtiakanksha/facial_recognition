{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "02438b00-e475-412e-b401-fc48df5b6095",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ! pip install flask tensorflow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "3c670741-3bcc-48b6-a7ed-6446323a4fc7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting opencv-python\n",
      "  Downloading opencv_python-4.10.0.84-cp37-abi3-win_amd64.whl.metadata (20 kB)\n",
      "Requirement already satisfied: numpy>=1.21.2 in c:\\users\\kosht\\anaconda3\\lib\\site-packages (from opencv-python) (1.26.4)\n",
      "Downloading opencv_python-4.10.0.84-cp37-abi3-win_amd64.whl (38.8 MB)\n",
      "   ---------------------------------------- 0.0/38.8 MB ? eta -:--:--\n",
      "   ---------------------------------------- 0.1/38.8 MB 1.7 MB/s eta 0:00:24\n",
      "   ---------------------------------------- 0.4/38.8 MB 5.0 MB/s eta 0:00:08\n",
      "   - -------------------------------------- 1.4/38.8 MB 10.9 MB/s eta 0:00:04\n",
      "   --- ------------------------------------ 3.0/38.8 MB 17.6 MB/s eta 0:00:03\n",
      "   ---- ----------------------------------- 4.3/38.8 MB 19.7 MB/s eta 0:00:02\n",
      "   ----- ---------------------------------- 5.3/38.8 MB 21.2 MB/s eta 0:00:02\n",
      "   ----- ---------------------------------- 5.8/38.8 MB 18.4 MB/s eta 0:00:02\n",
      "   -------- ------------------------------- 8.4/38.8 MB 23.3 MB/s eta 0:00:02\n",
      "   ----------- ---------------------------- 10.7/38.8 MB 31.2 MB/s eta 0:00:01\n",
      "   ------------ --------------------------- 12.4/38.8 MB 34.4 MB/s eta 0:00:01\n",
      "   -------------- ------------------------- 14.1/38.8 MB 34.4 MB/s eta 0:00:01\n",
      "   --------------- ------------------------ 15.5/38.8 MB 34.6 MB/s eta 0:00:01\n",
      "   ----------------- ---------------------- 16.8/38.8 MB 38.5 MB/s eta 0:00:01\n",
      "   ------------------ --------------------- 18.4/38.8 MB 36.4 MB/s eta 0:00:01\n",
      "   -------------------- ------------------- 19.5/38.8 MB 32.8 MB/s eta 0:00:01\n",
      "   --------------------- ------------------ 20.8/38.8 MB 31.2 MB/s eta 0:00:01\n",
      "   ---------------------- ----------------- 21.9/38.8 MB 29.7 MB/s eta 0:00:01\n",
      "   ----------------------- ---------------- 23.0/38.8 MB 27.3 MB/s eta 0:00:01\n",
      "   ------------------------- -------------- 24.3/38.8 MB 27.3 MB/s eta 0:00:01\n",
      "   -------------------------- ------------- 25.7/38.8 MB 26.2 MB/s eta 0:00:01\n",
      "   --------------------------- ------------ 27.0/38.8 MB 27.3 MB/s eta 0:00:01\n",
      "   ----------------------------- ---------- 28.3/38.8 MB 26.2 MB/s eta 0:00:01\n",
      "   ------------------------------ --------- 29.6/38.8 MB 27.3 MB/s eta 0:00:01\n",
      "   ------------------------------- -------- 30.9/38.8 MB 27.3 MB/s eta 0:00:01\n",
      "   --------------------------------- ------ 32.2/38.8 MB 27.3 MB/s eta 0:00:01\n",
      "   ---------------------------------- ----- 33.5/38.8 MB 27.3 MB/s eta 0:00:01\n",
      "   ----------------------------------- ---- 34.9/38.8 MB 28.4 MB/s eta 0:00:01\n",
      "   ------------------------------------- -- 36.2/38.8 MB 27.3 MB/s eta 0:00:01\n",
      "   -------------------------------------- - 37.6/38.8 MB 28.5 MB/s eta 0:00:01\n",
      "   ---------------------------------------  38.6/38.8 MB 28.4 MB/s eta 0:00:01\n",
      "   ---------------------------------------  38.8/38.8 MB 27.3 MB/s eta 0:00:01\n",
      "   ---------------------------------------- 38.8/38.8 MB 24.2 MB/s eta 0:00:00\n",
      "Installing collected packages: opencv-python\n",
      "Successfully installed opencv-python-4.10.0.84\n"
     ]
    }
   ],
   "source": [
    "# ! pip install opencv-python"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a8568fe3-e719-4d0e-98c1-9d33bc3bde75",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
      " * Running on http://127.0.0.1:5000\n",
      "INFO:werkzeug:\u001b[33mPress CTRL+C to quit\u001b[0m\n",
      "INFO:werkzeug: * Restarting with watchdog (windowsapi)\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, render_template, request, jsonify\n",
    "import base64\n",
    "import cv2\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load your trained model\n",
    "model = load_model('emotion_recognition_cnn.h5')\n",
    "\n",
    "@app.route('/')\n",
    "def index():\n",
    "    return render_template('index.html')  # Render your HTML file\n",
    "\n",
    "@app.route('/analyze-expression', methods=['POST'])\n",
    "def analyze_expression():\n",
    "    data = request.json\n",
    "    image_data = data['image'].split(',')[1]  # Extract base64 data\n",
    "    image = base64.b64decode(image_data)\n",
    "\n",
    "    # Convert the image to a numpy array\n",
    "    nparr = np.frombuffer(image, np.uint8)\n",
    "    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)\n",
    "    img = cv2.resize(img, (64, 64))  # Resize to model input shape\n",
    "    img = img.astype('float32') / 255.0  # Normalize\n",
    "    img = np.reshape(img, (1, 64, 64, 3))  # Reshape for model\n",
    "\n",
    "    # Predict emotion\n",
    "    prediction = model.predict(img)\n",
    "    mood = np.argmax(prediction)  # Get the index of the highest score\n",
    "\n",
    "    return jsonify({'mood': mood})  # Return mood index\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd053c6e-587d-4dc3-b598-62508d880be9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
