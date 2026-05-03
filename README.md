# Autonomous Surgical Task Failure Detection

This project utilizes deep learning architectures to automate the detection of five distinct failure modes in endoscopic surgical videos: **Collision, Task Completion, Smoke Detection, Incomplete Cut, and Tension**.

---

## Technical Approach
The pipeline followed a four-stage process to move from raw video to validated inference:

1.  **Data Acquisition & Annotation**: Images were extracted from endoscopic video frames and annotated across the five failure modes.
2.  **Overfit & Leakage Mitigation**: To address a generalization gap in initial trials, we implemented **randomized trial splits**, **residual auto-encoding**, and multiple **preprocessing** (normalization, resizing) techniques.
3.  **Model Benchmarking**: We trained and compared **ResNet, ViT, EfficientNet-B0**, and **EfficientNet-V2-M** to identify the optimal architecture for surgical failure detection.
4.  **Inference Testing**: Final validation was conducted on a holdout dataset using the best-performing model to ensure accuracy on unseen surgical data.

---

## Results Summary
The model achieved high reliability in safety-critical classification tasks, while temporal regression remains an area for future refinement.

| Failure Mode | Train Acc (%) | Val Acc (%) | Precision | Recall | Offline Inference Results |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Collision** | 99.78 | 85.10 | 0.84 | 0.92 | **96.31% Accuracy** |
| **Tension** | 99.80 | 86.23 | 0.94 | 0.78 | **86.00% Accuracy** |
| **Smoke Detection** | 99.82 | 100 | 1 | 1 | **84.96% Accuracy** |
| **Task Completion** | 84.09 | 45.00 | 0.66 | 0.52 | **53.01% Accuracy** |
| **Incomplete Cut** | 99.13 | 99.41 | 0.9934 | 0.9869 | **$r = 0.1326$ (Pearson)** |

---

## Future Work
*   **Expanded Dataset Training**: Improve model robustness and reduce generalization gaps by curating a larger, more diverse dataset of endoscopic frames.
*   **Multi-Modal Sensor Fusion**: Integrate robotic kinematic data with video features to provide physical context for tasks like **Tension** and **Incomplete Cut**.
*   **Lifelong Learning GUI**: Develop an interactive interface for clinicians to flag misclassifications in real-time, allowing for continuous model refinement on the server.


## Important: Repository & Data Constraints
**The source code in this repository cannot be executed locally.** 
*   **Data Size:** The endoscopic video datasets and annotation files are too large for GitHub storage. If needed, please contact Dr. Nural Yilmaz (nyilmaz2@jhu.edu) for access.
*   **Compute Environment:** All training and inference were performed by SSHing into the **jhu-aliss** server, which provided the necessary GPU resources for high-compute models like **EfficientNet-V2-M** and **Vision Transformers (ViT)**.
*   **Missing Artifacts:** Consequently, the trained model weights (`.pth` files), specific image labeling info, and large-scale annotation files are hosted externally on the JHU server and are not included in this repo. However, they can be accessed through this shared link: https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/msharm45_jh_edu/IgBU92QFDksHSpTzlIWHY7-YAXXK1FPmTwQ17FbG3dggjlA?e=HEvxGU 
