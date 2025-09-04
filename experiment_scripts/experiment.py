from pathlib import Path

from iris_wandml_project.pipeline.feature_preprocessing import FeatureProcessor
from iris_wandml_project.pipeline.data_preprocessing import DataProcessor
from iris_wandml_project.pipeline.model import ModelWrapper
from iris_wandml_project.config import Config
from experiment_scripts.evaluation import ModelEvaluator

DEFAULT_CONFIG = str(Path(__file__).parent / 'config.yaml')


class Experiment:
    def __init__(self):
        self._config = Config.from_yaml(DEFAULT_CONFIG)

    def run(self, train_dataset_path, test_dataset_path, output_dir, seed=42):
        return "results"