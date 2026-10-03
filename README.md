**CS 4372: Assignment 2:**

Dataset: DARWIN (handwriting data used to detect Alzheimer's disease) 
Team: Elizaveta Makhonina (dal445515) & Avizeh Walji (anw230000) 

**Files:**
assignment2.ipynb: all the code: loading the data, cleaning it up, picking features, training/tuning four models (Decision Tree, Random Forest, AdaBoost, XGBoost), and all the result plots.\
report_template.docx: the write-up. Has real numbers and plots from our own run already dropped in, plus little boxes marking where we still need to add our own thoughts before turning it in.\
DARWIN.csv: the dataset, exactly as we got it, to host on GitHub.


**How to run it:**
Open assignment2.ipynb in Colab.
Run all cells top to bottom. First cell installs xgboost; everything else is already in Colab.
The data loads from a GitHub raw link. RAW_CSV_URL in the second cell already points at our repo: https://raw.githubusercontent.com/elizardmak/4372_Assignment2/refs/heads/main/DARWIN.csv 


*Assumptions we made:*
No missing values, no duplicate rows in the raw data, so there was nothing to clean up there.
Every feature is a number already (times, speeds, pressure readings), so there was no categorical encoding to do.
Big one: this dataset has way more columns (450) than rows (174), so we had to pick a smaller set of features or the models would just overfit. We kept the top 20 by correlation with the target.
We split into train/test before picking features or scaling anything, and only used the training rows to decide which features to keep. Doing it the other way around (whole dataset first, then split) leaks test info into the model and makes results look better than they'd actually be. This was something we had to go back and fix after getting feedback on an earlier version.
80/20 split, stratified, random_state=42 — 139 training rows, 35 test rows. With that few test rows, small differences between models don't mean a whole lot, so don't read too much into a 2-3% gap.
StandardScaler is fit on training data only. Doesn't actually matter for any of our four models since trees don't care about feature scale, but it's in there since the assignment asks for it.
All four models tuned with GridSearchCV, 5-fold CV, scoring on F1.
AdaBoost's parameter for the base learner is named differently across scikit-learn versions (base_estimator used to be estimator), so the notebook checks the installed version and picks the right name automatically.
Dropped the ID column since it's just a participant number, not a real feature.
