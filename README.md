# Generalized_fuzzy_neural_network_public
This repository contains code related to the paper, **_Predicting Need for Heart Failure Advanced Therapies using an Interpretable Tropical Geometry-based Fuzzy Neural Network_**

@Author: 

- Yufeng Zhang chloezh@umich.edu
- Heming Yao

**Required Packages**
- torch==1.12.1
- numpy==1.21.6
- pandas==1.4.1
- scikit-learn==1.2.2
- matplotlib==3.5.1
- shap==0.40.0

### Code structure
1. Model
   * network
   * generalized fuzzy neural network
2. train_and_eval
   * train_test
   * cross-validation
   * rule extraction
3. data load
   * cannot be share the UM patient dataset due to data privacy
  
### Run the code
```
python main_cv.py
```
