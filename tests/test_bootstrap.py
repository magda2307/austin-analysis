import numpy as np
from unittest.mock import patch
from aac_adoption.models.bootstrap import bootstrap_ci

def test_bootstrap_preserves_repeated_cluster_draws():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    animal_ids = np.array(["A", "A", "B", "C"])

    def dummy_metric(y_t, y_p):
        # We can capture the length of y_t to see how many rows were passed
        return len(y_t)

    # We mock rng.choice to deterministically return ["A", "A", "B"]
    with patch("aac_adoption.models.bootstrap.np.random.default_rng") as mock_rng:
        mock_gen = mock_rng.return_value
        mock_gen.choice.return_value = np.array(["A", "A", "B"])

        # We need to compute bootstrap_ci with n_bootstraps=1
        lower, upper = bootstrap_ci(
            y_true=y_true,
            y_pred=y_pred,
            metric_func=dummy_metric,
            n_bootstraps=1,
            animal_ids=animal_ids
        )
        
        # In bootstrap_ci, unique_animals are "A", "B", "C" (len=3)
        # sampled_animals = ["A", "A", "B"]
        # "A" has indices [0, 1]. "B" has index [2]
        # sample_indices should be [0, 1, 0, 1, 2] -> length 5
        # metric_func returns the length of the sample, so score is 5
        # lower and upper bounds of [5] will both be 5.
        assert lower == 5.0
        assert upper == 5.0

def test_bootstrap_validates_input_lengths():
    y_true = np.array([1, 0])
    y_pred = np.array([1])
    animal_ids = np.array(["A", "B"])
    
    import pytest
    with pytest.raises(ValueError, match="Input lengths must match"):
        bootstrap_ci(
            y_true=y_true,
            y_pred=y_pred,
            metric_func=lambda x, y: 1,
            animal_ids=animal_ids
        )
