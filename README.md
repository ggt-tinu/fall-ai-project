# fall-ai-project
**Predicting High-Priced Airbnb Listings in NYC**

**Project Overview**

This project applies the machine learning life cycle to a real-world business problem: predicting whether a New York City Airbnb listing is "high-priced" (top 25% of listing prices) or "low-priced," using listing, host, and review features. It was completed as a capstone spanning traditional supervised learning and neural networks.

Two models were built and compared:

- A Decision Tree Classifier (traditional ML)
- A Feedforward Neural Network (Keras/TensorFlow)
  
**Objectives and Goals**
- Define a supervised, binary classification problem with clear business value.
- Explore and prepare a messy, real-world dataset (missing values, outliers, high-cardinality text columns, class imbalance).
- Train, tune, and evaluate a traditional ML model.
- Train and evaluate a neural network on the same problem.
- Compare both approaches on performance, interpretability, and practical deployment trade-offs, and reflect on the results.

**Dataset**


- Source: Airbnb NYC Listings data (airbnbListingsData.csv)
- Size: 28,022 listings, 51 original columns
- Label: price_category (low / high), high = listing price at or above the 75th percentile
- Class balance: ~75% low, ~25% high

A small synthetic sample_data.csv is included in this repo purely to illustrate the column schema and let others test the pipeline code. It is not real listing data, the original dataset is not redistributed here.

**Methodology**

Data Preparation:
- Flagged missingness with indicator columns, then imputed host_response_rate, host_acceptance_rate, bedrooms, and beds with column means.
- One-hot encoded neighbourhood_group_cleansed and room_type.
- Dropped free-text columns too high-cardinality to use without NLP (name, description, neighborhood_overview, host_name, host_location, host_about, amenities).
- Dropped zero-variance columns (host_is_superhost, host_has_profile_pic, host_identity_verified) and a redundant column (host_total_listings_count).
- Dropped price, since it directly determines the label and would leak the answer into the model.
  
Decision Tree
- Chose a Decision Tree for robustness to outliers and no need for feature scaling, given the extreme outliers found in features like minimum_nights.
- Used class_weight='balanced' to counter the ~75/25 class imbalance.
- Tuned max_depth and min_samples_leaf via GridSearchCV (5-fold, scoring on F1).
  
Neural Network
- Standardized all features with StandardScaler before training.
- Architecture: input layer → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, sigmoid).
- Optimizer: SGD, learning rate 0.01.
- Loss: binary cross-entropy.
- Trained 50 epochs with an 80/20 train/validation split.
  
**Results and Key Findings**

The Decision Tree achieved an accuracy of 0.830 and an F1 score of 0.616, while the Neural Network performed slightly better, with an accuracy of 0.850 and an F1 score of 0.667.

The neural network edged out the decision tree on both metrics, but the gap was small relative to the added complexity of building and tuning it.
The top predictive features for the Decision Tree are accommodates, room_type: Private room, neighbourhood_group_cleansed: Manhattan, bathrooms, and calculated_host_listings_count.
Training curves for the neural network showed mild overfitting (training loss kept falling while validation loss plateaued around epoch 20), but the train/validation accuracy gap stayed small, so the model still generalized reasonably well.

Recommendation: Deploy the Decision Tree. The performance difference is small, and the tree is far easier to explain to non-technical stakeholders, trains faster, and costs less to run than the neural network.

**Visualizations**


The notebook (Capstone.ipynb) includes:
- A class distribution countplot for price_category
- A histogram of accommodates split by price category
- A boxplot of minimum_nights for outlier detection
- A horizontal bar chart of the Decision Tree's top 10 feature importances
- Line plots of training vs. validation loss and accuracy over 50 epochs

**Potential Next Steps**
- Expand hyperparameter search ranges for both models (deeper grid for the Decision Tree; learning rate, epochs, and layer sizes for the neural network).
- Apply NLP techniques (e.g., TF-IDF, embeddings) to the text columns that were dropped (description, amenities, etc.) to recover their predictive signal.
- Try additional models (e.g., Random Forest, Gradient Boosting) for comparison.
- Investigate fairness metrics across neighbourhood_group_cleansed groups to quantify disparate error rates.
  
**Individual Contributions**


This was an individual capstone assignment; all analysis, modeling, and write-up were completed independently. AI assistance was used once, to debug a TensorFlow ValueError caused by boolean columns during Part 6 (neural network data prep) — the AI explained the root cause, and the fix (casting boolean columns to float) was implemented and verified independently.
