from freqres_pathology.models import FrequencyResidualClassifier, FrequencyResidualConfig, count_parameters
from freqres_pathology.eval.metrics import binary_metrics


def test_model_instantiation():
    model = FrequencyResidualClassifier(FrequencyResidualConfig())
    assert count_parameters(model) > 0


def test_binary_metrics():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)
    assert metrics["accuracy"] == 1.0
