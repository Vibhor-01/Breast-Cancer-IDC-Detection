# Breast-Cancer-IDC-Detection
Breast Cancer (IDC) Detection uses a CNN Deep learning model for detection of IDC in histological images, The model is Trained on histological images of 50x50 pixel size.  The model is a Contains 4 layers of Conv2D, and 4 Maxpooling2D layers, along with Flatten and Dense layers.

**Dataset Description:** The dataset used in the creation of our Cancer Detection System is carefully aggregated and scientifically validated. The dataset used is also used in the IEEE paper "A Deep Learning Approach for Breast Invasive Ductal Carcinoma Detection and Lymphoma
Multi-Classification in Histological Images" and has significantly contributed to the field of medical imaging. The paper serves as the backbone of our research and introduces a comprehensive collection of Breast histological images. The dataset contains histological images of the breast cancer tissues as well as healthy tissues of a patient. Each of these images are in a folder name as 0 and 1, where 1 contains the IDC images while the folder named as 0 contains the healthy tissue imagery (non-IDC). These folders in turn inside the folders of patients, with the folder names representing the patient id. The dataset is rich with diverse cases representing varying stages and manifestation of breast conditions. From a repository point of view, the dataset is inherently dynamic, and captures the variations that helps to distinguish healthy tissue from unhealthy tissue containing cancer lesions. Each image obtained through microscopy of tissues from biopsies, which brings the ability to observe tissue characteristics on a cell basis. 

**Model Description:** The Model is a CNN model with a validation accuray of 0.88 and validation loss of 0.32, this shows that the model being less complex than the models used for the same purpose offers more accuracy and less loss. Upon, comparison with other models, the proposed model stands out in terms of a blend between less complexity, more accuracy and low loss along with good precions and recall score.
