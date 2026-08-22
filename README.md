# Network Intrusion Detection System (NIDS) Framework

A machine learning framework for detecting and classifying network intrusions using the **NSL-KDD** benchmark dataset. This project pre-processes network traffic logs, handles class imbalance, and evaluates multi-class classification performance across various machine learning models.

---

## 📁 Project Structure

```text
nids-framework/
├── data/
│   ├── KDDTrain+.txt              # NSL-KDD Training Dataset
│   └── KDDTest+.txt               # NSL-KDD Testing Dataset
├── src/
│   ├── __init__.py                # Package initializer
│   ├── preprocessing.py           # Data loading, encoding, and scaling functions
│   ├── cgan_synthesis.py          # CGAN script for synthetic traffic generation
│   ├── deep_hybrid.py             # Deep learning / hybrid modeling pipeline
│   ├── ensemble_models.py         # Ensemble classification algorithms
│   └── explainability.py          # Model explainability utilities
├── confusion_matrix.png           # Auto-generated evaluation heatmap
├── main.py                        # Primary execution script
├── README.md                      # Project documentation
└── requirements.txt               # Dependencies list