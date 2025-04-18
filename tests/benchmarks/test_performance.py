import pytest


def method_x():
    # Your implementation
    pass


def method_y():
    # Alternative implementation
    pass


@pytest.mark.benchmark
def test_method_x(benchmark):
    benchmark(method_x)


@pytest.mark.benchmark
def test_method_y(benchmark):
    benchmark(method_y)
