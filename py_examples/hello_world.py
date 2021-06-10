import opengm


spaces = [
    opengm.UniformSpace(num_variables=10, num_labels=4),
    opengm.ExplicitSpace(num_variables=10, num_labels=3),
    opengm.BinarySpace(num_variables=10),
]

for space in spaces:
    print(
        f"""{space.name()}
            size {len(space)} space[0] {space[0]} max_num_labels {space.max_num_labels()} is_uniform {space.is_uniform_space()}
        """
    )


tensor = opengm.Potts2Tensor(2, 1.0)
print(tensor.dimension)
print(tensor.shape)
print(tensor[0, 1])
print(tensor[1, 0])
print(tensor[0, 1])


unary = opengm.UnaryTensor(2)
unary[1] = 1.0

print(unary[0], unary[1])


gms = [opengm.UniformSpaceGraphicalModel(num_variables=10, num_labels=2)]


for gm in gms:
    print(
        f"""
       {type(gm)} num variables {gm.num_variables()}
    """
    )

    tid = gm.add_tensor(opengm.Potts2Tensor(2, 1.0))
    fid = gm.add_factor(opengm.Potts2Tensor(2, 1.0), [0, 1])
    print("eval", gm.evaluate([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    print(tid)

    # self fusion
    self_fusion_factory = type(gm).SelfFusion.Factory(
        minimizer_factory=type(gm).BeliefPropergation.Factory(
            damping=1.0, num_iterations=100, convergence=1e-4
        ),
        fuse_minimizer_factory=opengm.FuseGm.FactorIcm.Factory(),
    )
    self_fusion = self_fusion_factory.create(gm)
    print("self_fusion ", self_fusion)

    # chained solvers
    chained_solvers_factory = type(gm).ChainedMinimizers.Factory(
        minimizer_factories=[
            type(gm).BeliefPropergation.Factory(),
            type(gm).FactorIcm.Factory(),
        ]
    )
    chained_solver = chained_solvers_factory.create(gm)
    print("chained_solver ", chained_solver)

    chained_solver.minimize()
