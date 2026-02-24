from src.steps.pipeline import GeneticAnalysisPipeline


def main():
    pipeline = GeneticAnalysisPipeline()
    pipeline.execute_full_pipeline()


if __name__ == "__main__":
    main()
