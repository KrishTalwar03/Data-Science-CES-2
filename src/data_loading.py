import pandas as pd
import os


class DataLoader:
    """Class to handle data loading and preprocessing"""

    def __init__(self, logger):
        """Initialize with logger"""
        self.logger = logger
        self.iris = None
        self.iris_df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
        self.csv_path = "dataset\\iris_species\\Iris.csv"

    def load_iris_dataset(self, csv_path=None):
        """
        Load the iris dataset from the Kaggle Iris.csv file

        Args:
            csv_path (str, optional): Path to the iris CSV file.
                If None, uses the default path set in the constructor.
        """
        self.logger.info("Step 2: Data Collection - Loading Iris dataset")

        # Define the default path if not provided
        if csv_path is None:
            csv_path = self.csv_path

        try:
            # Load the data from CSV
            self.logger.info(f"Loading data from: {csv_path}")
            self.iris_df = pd.read_csv(csv_path)
            
            # Map species to numeric target values (0, 1, 2)
            species_mapping = {
                'Iris-setosa': 0,
                'Iris-versicolor': 1,
                'Iris-virginica': 2
            }

            # Define feature column names from the actual CSV format
            self.feature_names = [
                'SepalLengthCm',
                'SepalWidthCm',
                'PetalLengthCm',
                'PetalWidthCm'
            ]

            # Get target names and map to integers
            species_col = 'Species'  # Column name from the actual CSV
            self.target_names = list(self.iris_df[species_col].unique())

            # Map species strings to integers
            self.iris_df['target'] = self.iris_df[species_col].map(species_mapping)

            # Extract features and target
            self.X = self.iris_df[self.feature_names].values
            self.y = self.iris_df['target'].values
            
            # Create a structure similar to what the rest of the code expects
            class IrisDataset:
                pass
            
            self.iris = IrisDataset()
            self.iris.data = self.X
            self.iris.target = self.y
            self.iris.feature_names = self.feature_names
            self.iris.target_names = self.target_names
            
            # Optional: Create more descriptive feature names like sklearn's
            self.iris.feature_names_descriptive = [
                'sepal length (cm)',
                'sepal width (cm)',
                'petal length (cm)',
                'petal width (cm)'
            ]
            
            self.logger.info("Successfully loaded iris dataset from CSV file")
            self.logger.info(f"Dataset shape: {self.iris_df.shape}")
            self.logger.info(f"First few rows:\n{self.iris_df.head()}")
            return self.iris, self.iris_df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}. Unable to load dataset.")
            raise

    def check_missing_values(self):
        """Check for missing values in the dataset"""
        missing_values = self.iris_df.isnull().sum().sum()
        self.logger.info(f"Total missing values: {missing_values}")
        return missing_values
