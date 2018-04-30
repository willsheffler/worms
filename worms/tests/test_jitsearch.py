import pytest
from worms.jitsearch import *

config = [
    ('C3_C', orient(None, 'C')),
    ('Het:NNC', orient('N', 'N')),
    ('Het:CN', orient('C', 'N')),
    ('Het:NNC', orient('C', None)),
]


@pytest.mark.skip
def test_jitsearch_0():
    print(config)
    assert 0