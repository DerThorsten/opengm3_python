import opengm
import numpy

numpy.random.seed(42)
shape = [15, 15]
num_variables = shape[0] * shape[1]
make_vi = lambda x, y: x * shape[1] + y
num_labels = 10
beta = 1.00
gm = opengm.UniformSpaceGraphicalModel(
    num_variables=num_variables, num_labels=num_labels
)

for x in range(shape[0]):
    for y in range(shape[1]):

        vi0 = make_vi(x, y)
        unary = numpy.random.rand(num_labels)
        gm.add_factor(unary, [vi0])

        if x + 1 < shape[0]:
            vi1 = make_vi(x + 1, y)
            gm.add_factor(opengm.Potts2Tensor(num_labels, beta), [vi0, vi1])
        if y + 1 < shape[1]:
            vi1 = make_vi(x, y + 1)
            gm.add_factor(opengm.Potts2Tensor(num_labels, beta), [vi0, vi1])

if True:
    factory = type(gm).SelfFusion.Factory(
        minimizer_factory=type(gm).BeliefPropergation.Factory(
            damping=0.1, num_iterations=1000, convergence=1e-4
        ),
        fuse_minimizer_factory=opengm.FuseGm.Icm.Factory(),
    )

else:
    factory = type(gm).BeliefPropergation.Factory(
        damping=0.9, num_iterations=10, convergence=1e-4
    )
minimizer = factory.create(gm)
# minimizer.set_starting_point(numpy.zeros(num_variables))
minimizer.minimize(type(gm).VerboseCallback(visit_nth=1))
best_labels = minimizer.best_labels().reshape(shape)
print(best_labels, "\n", minimizer.best_energy())
