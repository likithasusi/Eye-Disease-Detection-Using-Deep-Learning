# Eye Disease Detection Using Deep Learning

## System Requirements
- **Operating System:** Windows 8 or higher
- **Web Browsers:** Two installed web browsers
- **Internet Bandwidth:** Minimum 30 Mbps

## Project Description
This project aims to classify various types of eye diseases using deep learning techniques. Eye diseases can occur due to various factors, such as age, diabetes, and other medical conditions. The classification is done into four main categories:

1. **Normal**
2. **Cataract**
3. **Diabetic Retinopathy**
4. **Glaucoma**

Deep Learning (DL) techniques in Artificial Intelligence (AI) play a crucial role in high-performance classification tasks. By leveraging these advanced models, we can achieve accurate detection of eye diseases using medical images.

## Dataset Download
The dataset used for training and evaluation can be downloaded from Kaggle:

[Eye Diseases Classification Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

### Steps to Download the Dataset:
1. Sign in to your Kaggle account.
2. Navigate to the dataset link provided above.
3. Click on the **Download** button to get the dataset.
4. Extract the downloaded files and place them in the appropriate directory for training and testing.

## Methodology
### Transfer Learning
Transfer learning has become an essential technique in deep learning, especially for image analysis and classification. For this project, we utilize the following pre-trained models:

- **Inception V3**
- **VGG19**
- **Xception V3**

These models have been widely used in medical image classification and have demonstrated high effectiveness in recognizing complex patterns within images.

## Technical Architecture
1. **Dataset Preparation:**
   - Collect and preprocess medical images of eye diseases.
   - Perform data augmentation to improve model robustness.
   
2. **Model Selection & Training:**
   - Implement transfer learning using Inception V3, VGG19, and Xception V3.
   - Fine-tune pre-trained models to optimize classification accuracy.

3. **Evaluation & Testing:**
   - Use performance metrics such as accuracy, precision, recall, and F1-score.
   - Compare results of different models to select the best-performing one.

4. **Deployment:**
   - Develop a web-based interface for user-friendly interaction.
   - Integrate the trained model for real-time disease detection from uploaded images.

## Installation Guide
1. **Prerequisites:**
   - Install Python (>= 3.7)
   - Install required dependencies using:
     ```bash
     pip install -r requirements.txt
     ```
   - Install TensorFlow and Keras for deep learning:
     ```bash
     pip install tensorflow keras
     ```
   - Install OpenCV for image processing:
     ```bash
     pip install opencv-python
     ```

2. **Run the Application:**
   - Clone the repository:
     ```bash
     git clone <repository_link>
     ```
   - Navigate to the project folder:
     ```bash
     cd eye-disease-detection
     ```
   - Run the application:
     ```bash
     python app.py
     ```

## Results & Performance
- The project utilizes state-of-the-art deep learning models to classify eye diseases with high accuracy.
- Results will be presented with performance metrics and visualizations.

## Future Enhancements
- Expand dataset size to improve model generalization.
- Implement additional deep learning models for better accuracy.
- Develop a mobile application for remote diagnosis.

