import pytest

import opengm


class TestSpaces(object):
    @pytest.mark.parametrize(
        "space,",
        [
            opengm.UniformSpace(num_variables=3, num_labels=2),
            opengm.ExplicitSpace(num_variables=3, num_labels=2),
            opengm.BinarySpace(num_variables=3),
        ],
    )
    def test_space(self, space):
        assert len(space) == 3
        assert space[0] == 2
        assert space[1] == 2
        assert space[2] == 2

        assert space.max_num_labels() == 2
        assert space.is_uniform_space() == True


class TestGm(object):
    @pytest.mark.parametrize(
        "gm,",
        [
            opengm.UniformSpaceGraphicalModel(num_variables=3, num_labels=2),
            opengm.ExplicitSpaceGraphicalModel(num_variables=3, num_labels=2),
            opengm.BinarySpaceGraphicalModel(num_variables=3),
        ],
    )
    def test_gm(self, gm):
        assert gm.num_variables() == 3
        assert gm.num_labels(0) == 2
        assert gm.num_labels(1) == 2
        assert gm.num_labels(2) == 2
        space = gm.space()
        assert len(space) == 3
        assert space[0] == 2
        assert space[1] == 2
        assert space[2] == 2
        assert space.max_num_labels() == 2
        assert space.is_uniform_space() == True


class TestTensors(object):
    def test_potts2(self):
        tensor = opengm.Potts2Tensor(2, 1.0)

        assert tensor.shape == [2, 2]
        assert tensor.dimension == 2
        assert tensor[0, 0] == pytest.approx(0.0)
        assert tensor[0, 1] == pytest.approx(1.0)
        assert tensor[1, 0] == pytest.approx(1.0)
        assert tensor[1, 1] == pytest.approx(0.0)

    def test_unary(self):
        tensor = opengm.UnaryTensor(num_labels=3)

        assert tensor.shape == [3]
        assert tensor.dimension == 1
        assert tensor[0] == 0
        assert tensor[0] == 0
        assert tensor[0] == 0

        tensor[0] = 10.0
        tensor[1] = 11.0
        tensor[2] = 12.0

        assert tensor[0] == pytest.approx(10.0)
        assert tensor[1] == pytest.approx(11.0)
        assert tensor[2] == pytest.approx(12.0)

    def test_optimized_binary_unary(self):
        tensor = opengm.OptimizedBinaryUnary(val0=10, val1=9)

        assert tensor.shape == [2]
        assert tensor.dimension == 1
        assert tensor[0] == 1
        assert tensor[1] == 0

        tensor = opengm.OptimizedBinaryUnary(val0=10)
        assert tensor[0] == 10
        assert tensor[1] == 0
