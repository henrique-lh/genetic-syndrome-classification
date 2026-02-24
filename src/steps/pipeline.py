import logging

from .data_preprocessing import data_processing
from .data_visualization import data_visualization
from .classification import run_knn_classification
from .classification_visualization import run_knn_visualization
from .evaluation import run_evaluation

class GeneticAnalysisPipeline:
    """
    Facade class to execute the entire pipeline.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def run_preprocessing(self) -> None:
        self.logger.info("Step 1/5: Starting data processing...")
        data_processing()

    def run_eda(self) -> None:
        self.logger.info("Step 2/5: Starting data visualization (EDA)...")
        data_visualization()

    def run_classification(self) -> None:
        self.logger.info("Step 3/5: Starting KNN model training and selection...")
        run_knn_classification()

    def run_classification_visualization(self) -> None:
        self.logger.info("Step 4/5: Generating performance charts for classification...")
        run_knn_visualization()

    def run_evaluation(self) -> None:
        self.logger.info("Step 5/5: Final evaluation, metric calculation and ROC curves...")
        run_evaluation()

    def execute_full_pipeline(self) -> None:
        """
        Executes all pipeline steps sequentially.
        """
        self.logger.info("=== Starting Full Genetic Analysis Pipeline ===")
        try:
            self.run_preprocessing()
            self.run_eda()
            self.run_classification()
            self.run_classification_visualization()
            self.run_evaluation()
            self.logger.info("=== Pipeline executed successfully! ===")
        except Exception as e:
            self.logger.error(f"Fatal error during pipeline execution: {e}")
            raise
