import argparse
import logging
import sys
from src.steps.pipeline import GeneticAnalysisPipeline
from src.inference_service.app import app
import uvicorn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """Run the full analysis pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Genetic Analysis Pipeline")
    logger.info("=" * 60)

    try:
        pipeline = GeneticAnalysisPipeline()
        pipeline.execute_full_pipeline()
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        return True
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return False


def start_inference_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the inference server."""
    logger.info("=" * 60)
    logger.info(f"Starting Inference Server on {host}:{port}")
    logger.info("=" * 60)
    logger.info("API Documentation available at:")
    logger.info(f"  Swagger UI: http://localhost:{port}/docs")
    logger.info(f"  ReDoc: http://localhost:{port}/redoc")
    logger.info("=" * 60)

    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="Genetic Syndrome Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pipeline           # Run only the analysis pipeline
  python main.py --server             # Start only the inference server
  python main.py --all                # Run pipeline then start server (default)
  python main.py --server --port 9000 # Start server on port 9000
        """,
    )

    parser.add_argument(
        "--pipeline", action="store_true", help="Run only the analysis pipeline"
    )
    parser.add_argument(
        "--server", action="store_true", help="Start only the inference server"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run pipeline then start server (default behavior)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )

    args = parser.parse_args()

    run_pipeline_flag = args.pipeline
    run_server_flag = args.server
    run_all_flag = args.all

    if not (run_pipeline_flag or run_server_flag or run_all_flag):
        run_all_flag = True

    if run_pipeline_flag:
        success = run_pipeline()
        sys.exit(0 if success else 1)

    elif run_server_flag:
        start_inference_server(host=args.host, port=args.port)

    elif run_all_flag:
        success = run_pipeline()
        if success:
            logger.info("")
            logger.info("Pipeline execution completed. Starting inference server...")
            logger.info("")
            start_inference_server(host=args.host, port=args.port)
        else:
            logger.error("Pipeline failed. Exiting without starting server.")
            sys.exit(1)


if __name__ == "__main__":
    main()
