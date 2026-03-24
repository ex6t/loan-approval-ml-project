# Loan Approval Project

This is a beginner-to-advanced machine learning portfolio project based on a loan approval prediction system.

## Step 1 completed
- Project folder created
- Virtual environment created
- Packages installed
- Dataset downloaded
- Jupyter notebook working
- CSV loaded into pandas
- Initial data inspection completed

## Step 7 observations
- Built a standalone Python script for predicting one loan application.
- Loaded the trained Random Forest model and encoders from disk.
- Encoded categorical inputs exactly the same way as training.
- Returned both the predicted label and class probabilities.
- This script serves as the first bridge from notebook-based experimentation to deployable application logic.

## Step 8 observations
- Refactored prediction logic into a reusable function.
- Added multiple applicant test cases for stronger demo coverage.
- Added validation for categorical inputs.
- This structure is now ready to be reused inside an API endpoint.