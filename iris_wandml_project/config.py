from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml


class ConfigParsingFailed(Exception):
    pass


@dataclass
class DataConfig:
    version: str
    dataset_name: str
    feature_columns: List[str]
    target_column: str
    id_column: str


@dataclass
class FeaturesConfig:
    apply_scaling: bool
    scaling_method: str


@dataclass
class ModelEvalConfig:
    split_ratio: float
    cv_folds: int
    random_state: int
    evaluation_metric: str
    stratify: bool


@dataclass
class ModelConfig:
    model_type: str
    model_params: Dict[str, Any]
    random_state: int


@dataclass
class AlgorithmConfig:
    model_type: str
    base_params: Dict[str, Any]
    hyperparameter_grid: Dict[str, List[Any]]


@dataclass
class HyperparameterOptimizationConfig:
    method: str
    scoring: str
    cv_folds: int
    n_jobs: int


@dataclass
class Config:
    data_prep: DataConfig
    feature_prep: FeaturesConfig
    model_evaluation: ModelEvalConfig
    model: ModelConfig
    algorithms: Optional[Dict[str, AlgorithmConfig]] = None
    hyperparameter_optimization: Optional[HyperparameterOptimizationConfig] = None

    @staticmethod
    def from_yaml(config_file: str):
        with open(config_file, 'r', encoding='utf-8') as stream:
            try:
                config_data = yaml.safe_load(stream)
                
                # Parse algorithms if present
                algorithms = None
                if 'algorithms' in config_data:
                    algorithms = {}
                    for alg_name, alg_data in config_data['algorithms'].items():
                        algorithms[alg_name] = AlgorithmConfig(**alg_data)
                
                # Parse hyperparameter optimization if present
                hp_opt = None
                if 'hyperparameter_optimization' in config_data:
                    hp_opt = HyperparameterOptimizationConfig(**config_data['hyperparameter_optimization'])
                
                return Config(
                    data_prep=DataConfig(**config_data['data_prep']),
                    feature_prep=FeaturesConfig(**config_data['feature_prep']),
                    model_evaluation=ModelEvalConfig(**config_data['model_evaluation']),
                    model=ModelConfig(**config_data['model']),
                    algorithms=algorithms,
                    hyperparameter_optimization=hp_opt
                )
            except (yaml.YAMLError, OSError) as e:
                raise ConfigParsingFailed from e