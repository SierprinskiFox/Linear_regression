# Linear Regression using PyTorch
Linear regression model for the Hackathlon project. 02.2024
This project implements a linear regression model using PyTorch to predict a target variable (`Price Increase Ratio`) based on features (`Sentiment`, `Views`, and `Likes`). The implementation includes data preprocessing, model training, and evaluation.

## Features

- Handles missing or invalid values in the dataset.
- Scales input features using `StandardScaler` for improved performance.
- Uses PyTorch's `Dataset` and `DataLoader` for batch processing.
- Implements a training loop with optimization and learning rate scheduling.
- Evaluates model performance with Mean Squared Error (MSE) and R² score.

## File Structure

- **Code**: The main implementation is located in the Python file.
- **Data**: The code expects a CSV file (`../mongodb_data.csv`) as input, which must contain the following columns:
  - `Sentiment`
  - `Views`
  - `Likes`
  - `Price Increase Ratio` (target variable)

## Prerequisites

- Python 3.8+
- PyTorch
- Scikit-learn
- Pandas
- NumPy

To install the required Python packages, run:
```bash
pip install torch scikit-learn pandas numpy
```

## How to Run

1. Ensure the input data file (`../mongodb_data.csv`) is available in the specified path and formatted correctly.
2. Run the script:
   ```bash
   python <filename>.py
   ```

## Workflow

1. **Data Preprocessing:**
   - Missing or invalid values in the `Price Increase Ratio` column are replaced with the mean of the column.
   - The features (`Sentiment`, `Views`, and `Likes`) are scaled using `StandardScaler`.

2. **Model Training:**
   - A PyTorch `LinearRegressionModel` is trained using `MSELoss` as the loss function.
   - Stochastic Gradient Descent (SGD) with momentum is used for optimization.
   - Learning rate scheduling is applied to adjust the learning rate during training.

3. **Evaluation:**
   - The model is evaluated on the test set using Mean Squared Error (MSE) and R² score.

## Key Functions

- **RegressionDataset:** A custom PyTorch `Dataset` class to handle inputs and targets for training and testing.
- **Training Loop:** Includes forward pass, loss computation, and backpropagation with batch processing.
- **Evaluation Metrics:** Computes the test loss and R² score to evaluate model performance.

## Example Output

During training:
```
Epoch 10/100, Loss: 0.1234
Epoch 20/100, Loss: 0.0987
...
Epoch 100/100, Loss: 0.0456
```

After evaluation:
```
Test Loss: 0.0423
R² Score: 0.8912
```

## Notes

- Ensure the input data does not contain extreme outliers, as this may impact model performance.
- The model assumes a linear relationship between features and the target variable. If the relationship is non-linear, consider using a more complex model or feature engineering.

## License

This project is open-source and available under the MIT License.

## Contributing

Feel free to fork the repository and submit pull requests for any improvements or bug fixes. Suggestions for enhancing the model or code structure are welcome.

