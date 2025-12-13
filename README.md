# DDoS Attack Detection Using Supervised Machine Learning ⚡🛡️

## Project Overview
This project aims to detect **Distributed Denial-of-Service (DDoS) attacks** using **supervised machine learning** techniques. DDoS attacks are malicious attempts to overwhelm a network, server, or system with a flood of internet traffic, causing disruption of services. Detecting these attacks accurately is critical for maintaining the security and availability of networks.

The project is based on the **CICIDS2017 and CICIDS2018 datasets**, which provide realistic traffic data, including both normal and attack scenarios. By analyzing patterns in network traffic, the models can differentiate between normal and malicious behavior.

---

## Research Focus
The main objective of this research is to **improve upon previous studies** in DDoS detection by:  

1. **Feature Engineering:**  
   Creating new, meaningful features such as traffic rates, flow duration, packet size, and entropy measures to better capture patterns of attacks.

2. **Feature Selection:**  
   Using techniques like mutual information, recursive feature elimination, and feature importance scores to select the most informative features for the models.

3. **Handling Class Imbalance:**  
   Techniques such as SMOTE, ADASYN, and `class_weight='balanced'` are used to ensure minority attack classes are properly represented, improving model performance on rare attack types.

4. **Model Development & Ensembling:**  
   Building multiple models (tree-based, linear, distance-based) and combining them using ensemble methods like **voting** and **stacking** to improve detection accuracy and robustness.

5. **Cross-Dataset Testing:**  
   Testing models on a different dataset than the one used for training to evaluate generalization and real-world applicability.

6. **Evaluation Metrics:**  
   Comprehensive evaluation using **FPR, FNR, per-class F1 scores, ROC/AUC-PR curves**, ensuring realistic performance measurement.

7. **Explainability:**  
   Using methods like **SHAP** to understand the contribution of each feature in model predictions, providing insights into why certain traffic patterns are classified as attacks.

---

## Tools & Libraries 🛠️
- Python 3.x  
- pandas, numpy, scikit-learn  
- imbalanced-learn  
- XGBoost, LightGBM  
- SHAP  
- matplotlib, seaborn  

---

## Dataset 📂
- **CICIDS2017:** Primary training dataset with realistic normal and attack traffic.  
- **CICIDS2018:** Used for cross-dataset testing to validate model generalization.  

---

## Summary
This project combines advanced **data preprocessing, feature engineering, model development, and explainability techniques** to build an effective DDoS attack detection system. The methodology ensures models are not only accurate but also interpretable and robust against real-world variations in network traffic.
