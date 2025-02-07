# **Emotion Recognition using FER2013 Dataset**

![image](https://www.upwork.com/services/product/development-it-realtime-facial-emotion-recognition-detection-ai-web-or-mobile-app-1724687931832266752)
## **Introduction**
Emotion recognition is a significant task in computer vision that involves identifying human emotions based on facial expressions. This project leverages the FER2013 dataset to build a deep learning model capable of classifying images into seven emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**.

The project demonstrates the complete workflow of preprocessing the dataset, building and training a Convolutional Neural Network (CNN), evaluating its performance, and understanding the challenges and limitations of emotion recognition.

---

## **Project Goals**
1. Preprocess the FER2013 dataset, including resizing, normalization, and label encoding.
2. Visualize the dataset to better understand the data distribution and sample images.
3. Build and train a CNN for emotion classification.
4. Evaluate the model using metrics such as accuracy, confusion matrix, and classification report.
5. Analyze results and suggest future improvements.

---

## **Dataset Description**
The **FER2013 dataset** consists of grayscale facial images, each of size 48x48 pixels, labeled with one of seven emotions. The dataset is organized into training and testing directories, with subdirectories for each emotion.

### **Emotions and Labels**
- **Angry**: 0
- **Disgust**: 1
- **Fear**: 2
- **Happy**: 3
- **Sad**: 4
- **Surprise**: 5
- **Neutral**: 6

---

## **Techniques and Tools**
- **Deep Learning**: Convolutional Neural Networks (CNNs) for image processing.
- **Data Preprocessing**: Grayscale conversion, resizing, normalization.
- **Visualization**: Matplotlib and Seaborn for dataset exploration and training metrics visualization.
- **Data Augmentation**: Expands the dataset to improve generalization.
- **Evaluation Metrics**: Confusion matrix, classification report, and accuracy.
- **Frameworks and Libraries**:
  - TensorFlow/Keras
  - Pandas, NumPy
  - Matplotlib, Seaborn
  - PIL (Python Imaging Library)
  - Scikit-learn

---

## **Model Architecture**
We used a **Convolutional Neural Network (CNN)** with the following architecture:
1. **Convolutional Layers**:
   - Extract spatial features using filters.
   - Use batch normalization and dropout to prevent overfitting.
2. **Pooling Layers**:
   - Reduce spatial dimensions, retaining essential features.
3. **Dense Layers**:
   - Fully connected layers for final classification.

### Why CNN?
CNNs are highly effective for image data due to their ability to capture spatial hierarchies, such as edges and textures, and their scalability for complex features.

---

## **Evaluation**
### Outputs:
1. **Model Accuracy**:
   - Training and validation accuracy/loss curves plotted.
   - Test accuracy achieved: ~63%.
   
2. **Confusion Matrix**:
   - Visualization of correct vs. incorrect predictions.

3. **Classification Report**:
   - Precision, recall, and F1-score for each emotion.

---

## **Challenges and Limitations**
1. **Dataset Quality**:
   - Images are low-resolution and grayscale, making feature extraction harder.
2. **Class Imbalance**:
   - Emotions like "Disgust" have fewer samples, affecting model performance.
3. **Overlapping Features**:
   - Certain emotions share similar facial expressions, increasing misclassification.

---

## **Future Work**
To improve the model:
1. **Transfer Learning**: Utilize pre-trained models like ResNet or EfficientNet.
2. **Augmentation**: Apply more advanced data augmentation techniques.
3. **Class Balancing**: Use techniques like SMOTE or oversampling for underrepresented classes.
4. **Larger Dataset**: Incorporate more diverse and high-resolution datasets.

---

## **Conclusion**
This project provided insights into the complexity of emotion recognition. Although the achieved accuracy was modest, it lays the groundwork for further exploration and improvement. Emotion recognition remains an exciting field with vast applications in healthcare, psychology, and human-computer interaction.

---

## **Dataset**
[FER2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
