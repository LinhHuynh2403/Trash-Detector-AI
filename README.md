# ♻️ Trash Sorting AI (Recyclable vs Non-Recyclable)
UC Davis Computer Vision project that uses a pretrained CNN to classify waste (plastic, glass, paper, etc.) into recyclable vs. non-recyclable for promoting environmental sustainability.

## 🧠 Project Overview
This project aims to demonstrate how **AI can support environmental sustainability** by automatically classifying waste items (plastic, glass, paper, cardboard, metal, etc.).  
Instead of building a model from scratch, it uses **transfer learning** with pretrained models like **ResNet18** or **MobileNetV2** for efficient and accurate classification.

---

## 📊 Dataset
**Dataset:** [TrashNet](https://www.kaggle.com/code/lutfianaorinnijhum/trashnet-cnn/notebook)  
**Dataset:** [TrashNet V2](https://www.kaggle.com/code/akhileshs111111/trashnet-v2/input)  
- Contains 6 waste categories: *plastic, paper, metal, glass, cardboard, trash*  
- Used to train a model that outputs either:
  - 6-class waste type classification, or  
  - Simplified **recyclable vs non-recyclable** labels  

*(Note: Dataset not included in the repo due to size limits. Please download it from Kaggle.)*

---

## ⚙️ Methodology
1. **Data Preprocessing**
   - Resize and normalize images  
   - Apply data augmentation (rotation, flip, brightness)  

2. **Model Training**
   - Use a pretrained CNN (ResNet18 / MobileNetV2)  
   - Fine-tune the final layers for our dataset  
   - Train using categorical cross-entropy loss  

3. **Evaluation**
   - Accuracy and confusion matrix  
   - Grad-CAM visualization for interpretability  

4. **Output**
   - Classifies image as *Recyclable* or *Non-Recyclable*  

---
## Important Folders
1. **Results**
   - Includes all the results in json and csv file
   
2. **src**
   - training.py: handles data loading, model instanation (ResNet18), the training loop, printing epoch results, and saving all final assets (.csv, .pth, .json). Accepts --epochs and --lr arguments
   - utils.py: contains reusable components like the image pre-processing pipelines (val_transform), the function to build the ResNet18 model architecture (build_model), and the function to load the saved model checkpoint for inference (load_checkpoint_model)
   - test.py: loads the trained model from the results/folder and performs predictions on a list of individual images (defined by TEST_IMAGE_PATHS)
   - plot_metrics.py: loads the training metrics from results/training_metric.json and generates a plot visualizing the Training/Validation Loss and Accuracy over all epochs

3. **Results**
   - trash_sorting_finetuned_model.pth: Model Checkpoints. Contains the learned weights of the fine-tuned ResNet18 model
   - training_metric.csv: Metrics Log. A human-readable file logging the Loss, Accuracy, Epochs, Learning Rate for every epoch of the training run
   - training_metric.json: Plotting data. A JSON file containing the loss and accuracy lists, used as input for the plot_metrics.py
   - class_name.py: Class Mapping. A list of class name used to map the model's numerical output index (0, 1, 2,...) back to the human-readable garbage category. 

## 💻 How to Run
1. **Clone the Repo**
```bash
git clone https://github.com/LinhHuynh2403/Waste-Detector-AI.git
cd trash-sorting-ai
```
2. **Create a new virtual environment**
for Mac/Linux
```bash
python -m venv vevn
source venv/bin/activate   
``` 

for Windows (PowerShell)
```bash
python -m venv venv
venv\Scripts\Activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the training or test script**
```bash
python src/train.py     # for training
python src/predict.py   # for testing new images
```