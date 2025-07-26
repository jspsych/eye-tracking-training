Goal: Analyze how the number of unique participants in the training set impacts the model's performance.

The training dataset used in /models/densenet-regression-mask-trick-model.ipynb has about 1700 unique people in it. A separate test data set is used in /analysis/analysis-of-individual-model.run.ipynb. 

Plan:

- Load the training dataset and model architecture.
- Create function to sample subset of participants from the training set.
- Train the model on subsets of varying sizes (e.g., 10%, 20%, ..., 100% of unique participants).
- Evaluate the model's performance on the fixed test set.

Workflow:

- Log the training runs in Weights & Biases. 
- Store mean and standard deviation of performance metrics for each training run, as well as table of raw accuracy values. Save histogram of accuracy values for each training run, similar to what is already done in /analysis/analysis-of-individual-model.run.ipynb.
- Generate a plot showing the relationship between the number of unique participants in the training set and the model's performance metrics (e.g., accuracy, loss).