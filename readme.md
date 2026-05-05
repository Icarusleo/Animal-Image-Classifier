# 🐾 Animal Image Classification using DeiT

This is a Deep Learning (Image Classification) project aimed at classifying 90 different animal species using PyTorch and the `timm` library. The model utilizes the **DeiT (Data-efficient Image Transformers)** architecture via fine-tuning.

## 🌟 Features
- **Advanced Data Preprocessing:** Images are processed using Gaussian Blur and Histogram Equalization in the YUV color space via OpenCV.
- **Data Augmentation:** Techniques like random horizontal flip, rotation, and color jitter are applied to enhance the model's generalization capabilities.
- **Transformer-Based Model:** The `deit_base_patch16_224` model from the `timm` library is loaded with pre-trained ImageNet weights, and its final layer is adapted for 90 classes.
- **Comprehensive Metric Tracking:** Loss, Accuracy, Precision, Recall, and F1-Score metrics are calculated, tracked, and plotted during the training and validation phases.

## 🛠️ Technologies Used
- **Python 3**
- **PyTorch** & **Torchvision** (Deep Learning Framework)
- **timm** (PyTorch Image Models)
- **OpenCV** & **PIL** (Image Processing)
- **Scikit-learn** (Metric Calculations)
- **Matplotlib** (Data Visualization)

## 📂 Dataset
The project uses the **MultiZoo dataset** containing 90 different animal classes (e.g., lion, tiger, bear, dolphin, etc.). The dataset is included in the project as a `.zip` file via Google Drive and split into training/validation sets (80% - 20%).

## 🚀 Installation and Usage
1. Open a notebook on Google Colab and run this code.
2. Mount your Google Drive.
3. Ensure the dataset `.zip` file is located in the correct directory (`/content/drive/MyDrive/yazlab2.3/dataset/train.zip`).
4. Run the cells sequentially to install necessary libraries and train the model.

```bash
# Install the main required packages
pip install torch torchvision torchaudio
pip install timm opencv-python scikit-learn matplotlib
```

## 📈 Training Results and Outputs
During model training, the accuracy and loss values are printed at the end of each epoch. When the highest validation accuracy is achieved, the model weights are automatically saved to Drive as `best_deit_model_V2.pth`.
After the training process is complete, all metrics are written to the `training_history.json` file, and learning curves are plotted.
