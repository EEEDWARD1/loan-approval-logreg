import pandas as pd

class DataHandler:
    def __init__(self, file_name):
        """
        Initialize the DataHandler with the specified CSV file.

        :Param file_name: str
            The path to the CSV file containing the dataset.
        :Param data: pd.DataFrame
            The loaded dataset.
        :Returns: None
        """
        self.file_name = file_name
        self.data = None

    def load_data(self):
        """
        Load data from the CSV file into a pandas DataFrame.

        :Returns: None
        """
        try:
            self.data = pd.read_csv(self.file_name) 
        except FileNotFoundError:
            print(f"Error: The file {self.file_name} was not found.")
    
    def display_data(self, num_rows=5):
        """
        Display the first few rows of the dataset.

        :Param num_rows: int
            The number of rows to display from the dataset.
            
        :Returns: None
        """
        if self.data is not None:
            print(self.data.head(num_rows))
        else:
            print("No data loaded to display.")

    def format_data(self, target_column, categorical_columns=None):
        """
        Format the dataset by separating features and target variable,
        and encoding categorical variables if specified.

        :Param target_column: str
            The name of the target variable column.
        :Param categorical_columns: list of str
            The list of categorical variable column names to be encoded.

        :Returns: tuple (X, y)
            X: np.ndarray
                The feature matrix.
            y: np.ndarray
                The target variable array.
        """
        if self.data is None:
            print("No data loaded to format.")
            return None, None
        
        y = self.data[target_column].values
        X = self.data.drop(columns=[target_column])

        if categorical_columns:
            X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
            bools_col = X_encoded.select_dtypes(include=['bool']).columns
            X_encoded[bools_col] = X_encoded[bools_col].astype(float)
            return X_encoded.values, y
        
        return X.values, y

        