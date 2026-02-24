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
        self.logger.info("Passo 1/5: Iniciando processamento de dados...")
        data_processing()

    def run_eda(self) -> None:
        self.logger.info("Passo 2/5: Iniciando visualização de dados (EDA)...")
        data_visualization()

    def run_classification(self) -> None:
        self.logger.info("Passo 3/5: Treinamento e seleção do modelo KNN...")
        run_knn_classification()

    def run_classification_visualization(self) -> None:
        self.logger.info("Passo 4/5: Gerando gráficos de desempenho da classificação...")
        run_knn_visualization()

    def run_evaluation(self) -> None:
        self.logger.info("Passo 5/5: Avaliação final, cálculo de métricas e curvas ROC...")
        run_evaluation()

    def execute_full_pipeline(self) -> None:
        """
        Executa todos os passos do pipeline sequencialmente.
        """
        self.logger.info("=== Iniciando Pipeline Completo de Análise Genética ===")
        try:
            self.run_preprocessing()
            self.run_eda()
            self.run_classification()
            self.run_classification_visualization()
            self.run_evaluation()
            self.logger.info("=== Pipeline concluído com sucesso! ===")
        except Exception as e:
            self.logger.error(f"Erro fatal durante a execução do pipeline: {e}")
            raise
