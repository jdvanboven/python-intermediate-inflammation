from unittest.mock import Mock

def test_analyse_data_mock_source():
    from inflammation.compute_data import analyse_data
    data_source = Mock()
    data_source.load_inflammation_data.return_value = [
        [[1, 2, 3]],
        [[8, 12, 24]]
    ]

    analyse_data(data_source)