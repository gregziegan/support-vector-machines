import utils
import src.training as training

def test_get_cross_validation_sets():
    data = utils.get_test_data_set()
    cross_validation_sets = training.get_cross_validation_sets(data, 5)

    for cross_validation in cross_validation_sets:
        assert len(cross_validation['training_set']) == (4 * len(data)) / 5
        assert len(cross_validation['validation_set']) == (1 * len(data)) / 5