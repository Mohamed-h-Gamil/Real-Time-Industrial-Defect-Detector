# Real-Time Industrial Defect Detector (Edge-ML Prototype) 🏭

An end-to-end Computer Vision system for real-time quality control and industrial defect detection, designed with a hybrid Edge-to-Cloud architecture for low-latency performance in a simulated factory environment.

## 🌟 Project Highlights

* **Exceptional Performance:** Achieved an industry-grade **AUCROC of 0.99** using the PaDIM model for anomaly detection.
* **Hybrid Architecture:** Implemented a robust **Edge-to-Cloud pipeline**, leveraging an ESP32 as the data acquisition point.
* **Data Proficiency:** Architected and augmented a high-quality dataset of **1200+ images** from a minimal seed set (10 original images) to ensure model robustness.
* **Real-Time Monitoring:** Developed a dedicated frontend using Streamlit for live monitoring and inference visualization.

---

## 💻 Technical Stack

| Category | Tools & Libraries | Description |
| :--- | :--- | :--- |
| **Model** | **PaDIM** (Patch Distribution Modeling) | State-of-the-art anomaly detection model. |
| **Framework** | **Anomalib** (Intel) | Library used for rapid training, evaluation, and deployment of industrial anomaly models. |
| **Frontend/UI** | **Streamlit** | Developed a live web application for visualizing camera feed, model status, and inference results. |
| **Computer Vision** | **OpenCV** | Used for image processing, camera interfacing, and overlaying detection results. |
| **Data Handling** | **Torchvision Transforms | Python (NumPy, Pandas) | Used for automated dataset augmentation and manipulation. |
| **Edge Device** | **ESP32** | Simulated data acquisition endpoint (or actual hardware, depending on implementation). |

---

## 💡 Architecture: Edge-to-Cloud Pipeline

This project utilizes a hybrid architecture to balance data collection needs (Edge) with high-performance inference requirements (Cloud/Campus Server).

1.  **Edge Data Acquisition:** The **ESP32** (simulating a factory camera endpoint) captures images of bottle caps/items.
2.  **Transfer:** Images are sent to the central campus server hosting the application.
3.  **Inference (Cloud):** The trained **PaDIM** model runs on the server, performing the anomaly detection.
4.  **Real-Time Visualization:** Results are immediately sent to the **Streamlit** application frontend for operator monitoring.



---

## 📈 Performance & Results

The primary goal was to create a highly reliable defect detection system capable of distinguishing between normal and various abnormal item states.

* **Metric Achieved (AUCROC):** $\text{0.99}$
* **Dataset Size (Augmented):** $\text{1200+}$ images
* **Model Deployment:** Deployed via **Anomalib**, ready for containerization and scaling.

This high AUCROC score validates the model's robustness and efficiency for industrial quality assurance where minimizing false negatives (missed defects) is critical.

---

## 🛠️ Installation and Setup

### Prerequisites

* Python $\ge 3.8$
* `pip`
* Access to a device that can simulate or host the Streamlit server.

### Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Mohamed-h-Gamil/your-anomaly-repo.git](https://github.com/Mohamed-h-Gamil/your-anomaly-repo.git)
    cd your-anomaly-repo
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    The application will launch in your web browser, allowing you to interface with the model and visualize defect detection in real-time.

---

## 🤝 Contribution

This project was developed as a personal initiative to demonstrate practical MLOps and Computer Vision skills. Contributions and feedback are welcome!

---

**Developed by:** [Mohamed Gamil](https://www.linkedin.com/in/mohamed-gamil-5bba39233/)