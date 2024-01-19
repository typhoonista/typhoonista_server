import joblib
import numpy as np
from sklearn.tree import _tree

# Load the model from the file
model_filename = "STACKED-SVR_RF.joblib"  # Replace with the actual filename
loaded_model = joblib.load(model_filename)

# Define the expected dtype
expected_dtype = np.dtype([
    ('left_child', '<i8'),
    ('right_child', '<i8'),
    ('feature', '<i8'),
    ('threshold', '<f8'),
    ('impurity', '<f8'),
    ('n_node_samples', '<i8'),
    ('weighted_n_node_samples', '<f8'),
    ('missing_go_to_left', 'u1')
])

# Create a new tree with the expected dtype
new_tree = _tree.Tree(
    n_features=loaded_model.tree_.n_features,
    max_depth=loaded_model.tree_.max_depth,
    dtype=expected_dtype
)

# Copy relevant attributes from the loaded model to the new tree
new_tree.__setstate__(loaded_model.tree_.__getstate__())

# Create a new model with the new tree
new_model = type(loaded_model)(
    base_estimator=new_tree,
    n_estimators=len(loaded_model.estimators_),
    n_jobs=loaded_model.n_jobs
)

# Copy other attributes from the loaded model
new_model.__dict__.update(loaded_model.__dict__)

# Save the modified model
joblib.dump(new_model, "STACKED-SVR_RF_fixed.joblib", compress=True)



# # Example: Prepare input data for testing
# # Replace this with your actual input data
# input_data = [[1.0, 2.0, 3.0, 4.0,2.5,23,34]]
# input_data = np.array(input_data)
# # Reshape the 1D array to a 2D array
# input_data_2d = input_data.reshape(1, -1)

# # Make predictions using the loaded model

# # Make predictions using the loaded model
# predictions = loaded_model.predict(input_data_2d)

# # Display the predictions
# print("Predictions:", predictions)
# import sklearn
# print(sklearn.__version__)