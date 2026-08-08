# 🛍️ Olist E-Commerce Analytics
### End-to-End Data Science Project | Marketing · Logistics · NLP

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, Transformers, Matplotlib, Seaborn, SciPy
- **Database:** SQLite
- **Models:** KMeans (clustering) · Logistic Regression (NLP) · XLM-RoBERTa (transformer comparison)
- **NLP Pipeline:** TF-IDF + RSLPStemmer + Portuguese stopwords

---

## 📊 Key Findings

- **Time Series:** Revenue peaks in November (Black Friday). Delivery improved from 50+ to under 10 days. Late rate spikes during high demand.

- **RFM:** 4 customer segments — Champions, At Risk, Promising, Lost. Optimal clusters at k=4.

- **Logistics:** Carrier transit (9 days) is the bottleneck — 3x longer than vendor handling (2.7 days). Freight cost shows no clear linear relationship with distance — suggesting flat-rate pricing. SLA breach varies by state (p < 0.001). Late delivery drives negative sentiment.

- **NLP:** TF-IDF model achieves 76% accuracy — beats XLM-RoBERTa (67%). Neutral reviews are the biggest challenge (F1: 0.29).

- **Sentiment Drivers:** Negative = delivery/payment complaints; Neutral = mixed language; Positive = speed/quality praise.

**Conclusion:** A lightweight, interpretable model delivers better performance than deep learning for this task — proving that domain-specific feature engineering adds more value than model complexity.

---

## 📊 Visualizations

### 📈 Time Series & Business Performance

![Monthly Performance Dashboard](outputs/monthly_performance_dashboard.png)
*Revenue, orders, review scores, delivery days, and late rate trends over time.*

---

### 👥 Customer Segmentation

![RFM Scatter Plot](outputs/rfm_scatter.png)
*Customer segments by recency, frequency, and monetary value.*

![Elbow & Silhouette Validation](outputs/elbow_silhouette_validation.png)
*Optimal cluster selection for KMeans.*

---

### 🚚 Logistics Insights

![Distance vs Freight Cost & Bottleneck Analysis](outputs/distance_vs_freight.png)
*Left: Carrier transit (9 days) is 3x longer than vendor handling (2.7 days). Right: Flat-rate freight pricing confirmed.*

![Spearman Correlation: Drivers vs Sentiment](outputs/spearman_correlation_drivers_vs_sentiment.png)
*Logistics features correlated with review sentiment.*

![SLA Breach by State](outputs/sla_breach_by_state.png)
*Regional delivery performance — SLA breach varies significantly by state (p < 0.001).*

---

### 💬 NLP & Sentiment Analysis

![NLP Dashboard](outputs/nlp_dashboard.png)
*Review length, score distribution, and key word frequencies by delivery status.*

![Predictive Words WordCloud](outputs/predictive_words_wordcloud_colored.png)
*Most predictive words for negative (red), neutral (blue), and positive (green) sentiment.*

![Predictive Words Bar Chart](outputs/predictive_words_barchart.png)
*Top 10 words driving each sentiment class.*

---

## 📥 Installation

**Prerequisites:** Python 3.10+

```bash
# Clone repository
git clone https://github.com/aguchhait-stack/olist-supply-chain-analytics.git
cd olist-supply-chain-analytics
 
# Install dependencies
pip install -r requirements.txt

# Windows (if required)
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install wordcloud --prefer-binary

# Launch notebook
jupyter notebook notebooks/01_olist_supply_chain_analysis.ipynb

# Run all tests
pytest tests/ -v
```
---

## 🚧 Current Development Status

The repository is currently being refactored from a research notebook into a production-grade Python package.

---

## 📚 Data Citation
Olist, and André Sionek. (2018). Brazilian E-Commerce Public Dataset by Olist [Dataset]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/195341
