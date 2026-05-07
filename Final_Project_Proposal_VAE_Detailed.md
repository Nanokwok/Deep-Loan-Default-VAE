# Deep Loan Default Prediction via VAE-based Anomaly Detection

**Course:** 204466 Deep Learning  
**Project Status:** Detailed Project Proposal  
**Date:** May 2026

---

## 1. Project Topic
**Deep Loan Default Prediction using Variational Autoencoders (VAE) as an Anomaly Detection Mechanism.**

## 2. Why is this topic interesting?
Loan default prediction is a cornerstone of risk management in financial institutions. The ability to accurately identify potential defaulters directly impacts a bank's profitability and financial stability. 

The primary challenge in this domain is **Extreme Data Imbalance**. In a typical lending environment, the vast majority of borrowers are "Normal" (repay their loans), while "Defaulters" (those who fail to pay) constitute a tiny minority. Conventional supervised learning models often struggle with this skewness, potentially leading to high false-negative rates where risky borrowers are incorrectly classified as safe. This project proposes using an unsupervised approach to model the distribution of "normal" credit behavior and flag deviations as potential defaults.

## 3. Why Deep Learning? (Comparison & Trade-offs)
Deep Learning, specifically Variational Autoencoders (VAEs), offers a unique paradigm compared to traditional methods like XGBoost or Random Forest.

### Strengths of the Deep Learning Approach (VAE):
* **Unsupervised Anomaly Detection:** Unlike standard classifiers, a VAE can be trained exclusively on "Successful Repayment" data to learn the underlying distribution of a "good borrower." Defaulters are then identified by their high **Reconstruction Error**, as the model will struggle to reconstruct patterns it has never seen.
* **Non-linear Latent Space Representation:** VAEs can perform non-linear dimensionality reduction, capturing hidden manifold structures in complex financial data that linear PCA or tree-based splits might overlook.
* **Robustness to Imbalance:** By shifting the focus from classification to distribution modeling, we mitigate the bias caused by the scarcity of default samples.

### Weaknesses vs. Traditional Approaches:
* **Interpretability:** Tree-based models (XGBoost) provide clear feature importance and decision paths. VAEs are "black-box" models, making it harder to explain *why* a specific loan was flagged as risky to stakeholders.
* **Complexity & Hyperparameters:** VAEs require careful tuning of the latent space dimension and the balance between Reconstruction Loss and KL Divergence ($\\beta$-VAE).
* **Data Requirements:** Deep learning typically requires more samples to converge effectively compared to traditional machine learning on tabular data.

## 4. Deep Learning Architecture
The model will follow a standard Variational Autoencoder architecture designed for tabular data:

| Component | Detailed Technical Description |
| :--- | :--- |
| **Input Layer** | Accepts preprocessed financial features (e.g., debt-to-income ratio, annual income, credit scores). |
| **Encoder** | A series of fully connected (Dense) layers with ReLU activation. It maps the input $x$ to two vectors: $\\mu$ (mean) and $\\sigma$ (variance) representing a Gaussian distribution in the latent space. |
| **Bottleneck (Z)** | Implements the **Reparameterization Trick**: $z = \\mu + \\sigma \\cdot \\epsilon$, where $\\epsilon \\sim N(0, 1)$. This allows for gradient flow during backpropagation. |
| **Decoder** | Mirroring the Encoder, it takes the sampled $z$ and attempts to reconstruct the original input features $\\hat{x}$. |
| **Loss Function** | **Total Loss = Reconstruction Loss (MSE) + $\\beta \\cdot$ KL Divergence.** The MSE ensures the model learns the data structure, while KL Divergence regularizes the latent space toward a standard normal distribution. |

## 5. Dataset and Source
**Dataset:** [Lending Club Accepted Loans Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)  
**Type:** Tabular data containing borrower profiles, loan characteristics, and payment history.

### Crucial Data Preprocessing & Leakage Prevention:
To ensure the model is valid for real-world prediction, we will strictly enforce a "Point-in-Time" data split:
* **Feature Selection:** We will only use features available **at the time of the loan application** (e.g., `loan_amnt`, `term`, `int_rate`, `annual_inc`, `dti`, `fico_range_low`).
* **Data Leakage Removal:** All columns generated *after* loan approval (e.g., `total_pymnt`, `recoveries`, `last_pymnt_amnt`, `out_prncp`) will be **dropped**. Including these would allow the model to "cheat" by seeing future payment success.
* **Target Definition:** We will filter for completed loans: `Fully Paid` (Normal) vs. `Charged Off` (Anomaly).

## 6. Training Method
1.  **Normalization:** Standardize numerical features and apply One-Hot Encoding to categorical variables.
2.  **Semi-Supervised Setup:** Train the VAE only on the `Fully Paid` class. This forces the model to become an "expert" at representing good credit behavior.
3.  **Optimizer:** Adam optimizer with a decaying learning rate.
4.  **Batch Size:** 512 or 1024 to handle the large Lending Club dataset efficiently.

## 7. Evaluation Metrics
Since this is an anomaly detection task, standard "Accuracy" is misleading. We will use:
* **Reconstruction Error Thresholding:** Determining the optimal cutoff point to separate Normal from Default.
* **Precision-Recall Curve & AUPRC:** The primary metric for imbalanced data performance.
* **Recall (Sensitivity):** Crucial for banks to catch as many potential defaulters as possible, even if some safe borrowers are flagged for manual review.
* **F1-Score:** To find the balance between catching defaults and maintaining precision.

## 8. Reference Articles
1.  Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*.
2.  An, J., & Cho, S. (2015). *Variational Autoencoder based Anomaly Detection using Reconstruction Probability*.
3.  Lending Club Open Data Documentation.

---
**Required Submission Materials:**
* `final_report.pdf` (Detailed analysis and diagrams)
* `README.md` (Based on this proposal)
* GitHub Repository with PyTorch implementation.
