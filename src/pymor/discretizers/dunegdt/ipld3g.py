import numpy as np

from pymor.algorithms.preassemble import preassemble as preassemble_
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.algorithms.timestepping import ExplicitEulerTimeStepper, ImplicitEulerTimeStepper
from pymor.bindings.dunegdt import DuneXTMatrixOperator
from pymor.discretizers.dunegdt.problems import StationaryDuneProblem
from pymor.discretizers.dunegdt.cg import _discretize_stationary_cg_dune
from pymor.discretizers.dunegdt.ipdg import (
    _discretize_stationary_ipdg_dune, _IP_estimate_penalty_parameter, _IP_scheme_id)
from pymor.models.basic import InstationaryModel, StationaryModel
from pymor.operators.block import BlockOperator, BlockColumnOperator
from pymor.operators.constructions import LincombOperator

from dune.xt.grid import (
    AllNeumannBoundaryInfo,
    ApplyOnCustomBoundaryIntersections,
    CouplingIntersection,
    Dim,
    DirichletBoundary,
    Cube,
    Walker,
    make_cube_dd_grid,
)
from dune.xt.functions import GridFunction as GF
from dune.xt.la import Istl, IstlVector
from dune.gdt import (
    BilinearForm,
    LocalIntersectionIntegralBilinearForm,
    LocalCouplingIntersectionIntegralBilinearForm,
    LocalLaplaceIPDGInnerCouplingIntegrand,
    LocalLaplaceIPDGDirichletCouplingIntegrand,
    LocalIPDGInnerPenaltyIntegrand,
    LocalIPDGBoundaryPenaltyIntegrand,
    MatrixOperator,
    make_coupling_sparsity_pattern,
)


def discretize_stationary_ipld3g(
    analytical_problem,
    macro_diameter=None,
    num_local_refinements=None,
    order=1,
    data_approximation_order=2,
    la_backend=Istl(),
    symmetry_factor=1,
    weight_parameter=None,
    penalty_parameter=None,
    preassemble=True,
    locally_continuous=True
):
    '''
    interior penalty localized domain decomposition discontinuous Galerkin = IPLDDDG = IPLD3G
    '''

    assert isinstance(analytical_problem, StationaryProblem)
    assert analytical_problem.advection is None, "Not implemented yet!"
    assert analytical_problem.diffusion is not None, "Not implemented yet!"
    assert analytical_problem.robin_data is None, "Not implemented yet!"
    assert analytical_problem.neumann_data is None, "Not implemented yet!"
    assert analytical_problem.dirichlet_data is None, "Not implemented yet!"
    assert analytical_problem.outputs is None, "Not implemented yet!"

    # we only use the macro problem for the macro grid, the boundary info and sanity checks
    if macro_diameter is None:
        macro_diameter = analytical_problem.domain.dim/(3*np.sqrt(2))
    macro_dune_problem = StationaryDuneProblem.from_pymor(
        analytical_problem, 0, diameter=macro_diameter, grid_type='cube')
    # TODO: enable 'simplex' as macro and local grid_type
    macro_grid, macro_boundary_info = macro_dune_problem.grid, macro_dune_problem.boundary_info
    del macro_dune_problem
    d = macro_grid.dimension

    if num_local_refinements is None:
        num_local_refinements = 5
    dd_grid = make_cube_dd_grid(macro_grid, local_element_type=Cube(), num_refinements=num_local_refinements)

    # we obtain the discrete model by
    # - collecting the coupling contributions in a separate BlockOperator
    M = dd_grid.num_subdomains
    local_ops = np.empty((M, M), dtype=object)
    local_rhs = np.empty((M, 1), dtype=object)
    coupling_ops = np.empty((M, M), dtype=object)
    weighted_h1_semi_penalty_product_ops = np.empty((M, M), dtype=object)

    # - building local models from discretizations of localized problems
    #   without essential boundary conditions
    #   TODO: adapt for non-trivial Neumann and/or Robin boundary data
    local_analytical_problem = analytical_problem.with_(dirichlet_data=None)
    #   * convert/interpolate data functions on local grids
    local_problems = [StationaryDuneProblem.from_pymor(
        local_analytical_problem, data_approximation_order,
        grid=dd_grid.local_grid(I), boundary_info=AllNeumannBoundaryInfo(dd_grid.local_grid(I)))
        for I in range(M)]
    #   * discretize locally
    local_models = []
    local_models_data = []
    local_spaces = []
    for I in range(M):
        if locally_continuous:
            local_model, local_model_data = _discretize_stationary_cg_dune(
                local_problems[I], order=order, la_backend=la_backend, preassemble=preassemble)
        else:
            local_model, local_model_data = _discretize_stationary_ipdg_dune(
                local_problems[I], order=order, la_backend=la_backend, symmetry_factor=symmetry_factor,
                weight_parameter=weight_parameter, penalty_parameter=penalty_parameter,
                preassemble=preassemble)
        local_models.append(local_model)
        local_models_data.append(local_model_data)
        local_spaces.append(local_model_data['space'])
        if isinstance(local_model.operator, LincombOperator):
            local_op = local_model.operator.with_(
                operators=[op.with_(name=f'volume_part_{I}')
                           for op in local_model.operator.operators], name='')
        else:
            local_op = local_model.operator.with_(name=f'volume_part_{I}')
        local_ops[I][I] = local_op
        local_rhs[I] = local_model.rhs.with_(name='')

    #   (from here on, we basically follow discretize_stationary_ipdg)
    IP_scheme_ID = _IP_scheme_id(symmetry_factor, weight_parameter)  # performs some checks
    # penalty parameter for the diffusion part of the IPDG scheme
    if penalty_parameter is None:
        # TODO: add missing min diffusion estimate, see discretize_stationary_ipdg
        if locally_continuous:
            for I in range(M):
                local_models_data[I]['IP_penalty_parameter'] = _IP_estimate_penalty_parameter(
                    local_problems[I].grid, local_models_data[I]['space'],
                    symmetry_factor, weight_parameter)
        penalty_parameter = np.max([data['IP_penalty_parameter'] for data in local_models_data])
        print(f'penalty parameter is {penalty_parameter}')
    # weight for the diffusion part of the IPDG scheme (see above)
    if weight_parameter is None:
        if locally_continuous:
            local_weights = [GF(p.grid, 1, (Dim(d), Dim(d))) for p in local_problems]
        else:
            local_weights = [data['IP_weight'] for data in local_models_data]
    else:
        mu_weight = local_problems[0].diffusion.parameters.parse(weight_parameter)
        local_weights = [p.diffusion.assemble(mu_weight) for p in local_problems]

    # - weak enforcing of Dirichlet boundary values
    for I in dd_grid.boundary_subdomains:
        local_grid = local_problems[I].grid
        walker = Walker(local_grid)
        boundary_info = dd_grid.macro_based_boundary_info(I, macro_boundary_info)

        def make_boundary_contributions_parametric_part(func):
            bf = BilinearForm(local_grid)
            bf += (LocalIntersectionIntegralBilinearForm(
                LocalLaplaceIPDGDirichletCouplingIntegrand(
                    symmetry_factor,
                    GF(local_grid, func, (Dim(d), Dim(d))))),
                   ApplyOnCustomBoundaryIntersections(
                       local_grid,
                       boundary_info,
                       DirichletBoundary()))
            op = MatrixOperator(local_grid, local_models_data[I]['space'],
                                local_models_data[I]['space'], la_backend,
                                local_models_data[I]['sparsity_pattern'])
            op.append(bf)
            walker.append(op)
            return op

        def make_boundary_contributions_nonparametric_part():
            bf = BilinearForm(local_grid)
            bf += (LocalIntersectionIntegralBilinearForm(
                LocalIPDGBoundaryPenaltyIntegrand(
                    penalty_parameter,
                    local_weights[I])),
                   ApplyOnCustomBoundaryIntersections(
                       local_grid,
                       boundary_info,
                       DirichletBoundary()))
            op = MatrixOperator(local_grid, local_models_data[I]['space'],
                                local_models_data[I]['space'], la_backend,
                                local_models_data[I]['sparsity_pattern'])
            op.append(bf)
            walker.append(op)
            return op

        ops = [make_boundary_contributions_parametric_part(func)
               for func in local_problems[I].diffusion.functions] + \
            [make_boundary_contributions_nonparametric_part(), ]
        coeffs = list(local_problems[I].diffusion.coefficients) + [1., ]

        walker.walk(False)  # not supported yet

        boundary_operators = [DuneXTMatrixOperator(op.matrix, name=f'boundary_part_{I}')
                              for op in ops]
        # give the constant part a special name
        boundary_operators[-1] = boundary_operators[-1].with_(name=f'constant_boundary_part_{I}')

        boundary_op = LincombOperator(operators=boundary_operators, coefficients=coeffs)

        local_ops[I][I] += boundary_op

    lhs_without_coupling = BlockOperator(local_ops)

    # - coupling of the local models by IP techniques
    for I in range(M):
        for J in dd_grid.neighbors(I):
            if I < J:  # treat each coupling only once, but from both sides
                coupling_grid = dd_grid.coupling_grid(I, J)
                walker = Walker(coupling_grid)
                coupling_sparsity_pattern = make_coupling_sparsity_pattern(
                    local_models_data[I]['space'],
                    local_models_data[J]['space'],
                    coupling_grid)

                def make_coupling_ops_from_bilinear_form(bf):
                    op_I_I = MatrixOperator(coupling_grid, local_models_data[I]['space'],
                                            local_models_data[I]['space'], local_models_data[I]['sparsity_pattern'])
                    op_I_J = MatrixOperator(coupling_grid, local_models_data[I]['space'],
                                            local_models_data[J]['space'], coupling_sparsity_pattern)
                    # TODO: transpose pattern?!
                    op_J_I = MatrixOperator(coupling_grid, local_models_data[J]['space'],
                                            local_models_data[I]['space'], coupling_sparsity_pattern)
                    op_J_J = MatrixOperator(coupling_grid, local_models_data[J]['space'],
                                            local_models_data[J]['space'], local_models_data[J]['sparsity_pattern'])
                    # volume, in_in, in_out, out_in, out_out, boundary
                    op_I_I.append(bf, {}, (False, True, False, False, False, False))
                    op_I_J.append(bf, {}, (False, False, True, False, False, False))
                    op_J_I.append(bf, {}, (False, False, False, True, False, False))
                    op_J_J.append(bf, {}, (False, False, False, False, True, False))
                    walker.append(op_I_I)
                    walker.append(op_I_J)
                    walker.append(op_J_I)
                    walker.append(op_J_J)
                    return op_I_I, op_I_J, op_J_I, op_J_J

                def make_coupling_contributions_parametric_part(func_in, func_out):
                    bf = BilinearForm(coupling_grid)
                    bf += LocalCouplingIntersectionIntegralBilinearForm(
                        LocalLaplaceIPDGInnerCouplingIntegrand(
                            symmetry_factor,
                            GF(local_problems[I].grid, func_in, (Dim(d), Dim(d))),
                            GF(local_problems[I].grid, func_out, (Dim(d), Dim(d))),
                            local_weights[I],
                            local_weights[J],
                            intersection_type=CouplingIntersection(dd_grid)))
                    return make_coupling_ops_from_bilinear_form(bf)

                def make_coupling_contributions_nonparametric_part():
                    bf = BilinearForm(coupling_grid)
                    bf += (LocalCouplingIntersectionIntegralBilinearForm(
                        LocalIPDGInnerPenaltyIntegrand(
                            penalty_parameter,
                            local_weights[I],
                            local_weights[J],
                            intersection_type=CouplingIntersection(dd_grid))))
                    return make_coupling_ops_from_bilinear_form(bf)

                ops_list = [[], [], [], []]
                coeffs = []
                for diff_in, coeff_in, diff_out, coeff_out in zip(
                    local_problems[I].diffusion.functions,
                    local_problems[I].diffusion.coefficients,
                    local_problems[J].diffusion.functions,
                    local_problems[J].diffusion.coefficients
                ):
                    assert coeff_in == coeff_out
                    ops_list = [ops + [additional_ops] for ops, additional_ops in zip(
                                ops_list, make_coupling_contributions_parametric_part(diff_in, diff_out))]
                    coeffs.append(coeff_in)
                ops_list = [ops + [additional_ops] for ops, additional_ops in zip(
                            ops_list, make_coupling_contributions_nonparametric_part())]
                ops_I_I, ops_I_J, ops_J_I, ops_J_J = ops_list[0], ops_list[1], ops_list[2], ops_list[3]
                del ops_list
                coeffs.append(1.)

                local_weighted_h1_semi_penalty_product_ops = make_coupling_contributions_nonparametric_part()

                walker.walk(False)  # parallel assembly not yet supported

                for (i, j, ops) in ((I, I, ops_I_I), (I, J, ops_I_J), (J, I, ops_J_I), (J, J, ops_J_J)):
                    operators = [DuneXTMatrixOperator(op.matrix, name=f'coupling_part_from_{I}_{J}')
                                 for op in ops]
                    operators[-1] = operators[-1].with_(name=f'constant_coupling_part_from_{I}_{J}')
                    coupling_op = LincombOperator(operators=operators, coefficients=list(coeffs))
                    if coupling_ops[i][j] is None:
                        coupling_ops[i][j] = coupling_op
                    if local_ops[i][j] is None:
                        local_ops[i][j] = coupling_op
                    else:
                        coupling_ops[i][j] += coupling_op
                        local_ops[i][j] += coupling_op

                for ((i, j), op) in zip(((I, I), (I, J), (J, I), (J, J)), local_weighted_h1_semi_penalty_product_ops):
                    coupling_op = DuneXTMatrixOperator(op.matrix, name=f'coupling_part_from_{I}_{J}')
                    if weighted_h1_semi_penalty_product_ops[i][j] is None:
                        weighted_h1_semi_penalty_product_ops[i][j] = LincombOperator(
                            operators=[coupling_op, ],
                            coefficients=[1., ])
                    else:
                        weighted_h1_semi_penalty_product_ops[i][j] += coupling_op

    # products
    local_l2_ops = np.empty((M, M), dtype=object)
    local_l2_0_ops = np.empty((M, M), dtype=object)
    # - assemble subdomain contributions
    for I in range(M):
        local_l2_ops[I][I] = local_models[I].products['l2']
        local_l2_0_ops[I][I] = local_models[I].products['l2_0']
        if 'weighted_h1_semi_penalty' in local_models[I].products:
            local_weighted_h1_semi_penalty_prod = local_models[I].products['weighted_h1_semi_penalty']
        else:
            local_weighted_h1_semi_penalty_prod = local_models[I].products['h1_semi']
        # entry I, I has to exist after the assembly above
        weighted_h1_semi_penalty_product_ops[I][I] += local_weighted_h1_semi_penalty_prod
        local_op = local_ops[I][I]
        for op in local_op.operators:
            if 'constant_boundary_part' in op.name:
                weighted_h1_semi_penalty_product_ops[I][I] += op
    products = {
        'l2': BlockOperator(local_l2_ops),
        'l2_0': BlockOperator(local_l2_0_ops),
        'weighted_h1_semi_penalty': BlockOperator(weighted_h1_semi_penalty_product_ops),
        'h1': BlockOperator(local_l2_ops) + BlockOperator(weighted_h1_semi_penalty_product_ops)
        # check whether this is the h1 product
    }

    coupling_op = BlockOperator(coupling_ops)
    lhs_op = BlockOperator(local_ops)
    rhs_op = BlockColumnOperator(local_rhs)
    m = StationaryModel(lhs_op, rhs_op, products=products,
                        name=f'{analytical_problem.name}_P{order}{IP_scheme_ID[:-2]}LD3G')

    # Fake weak Dirichlet handling for localized residual
    fake_dirichlet_ops = np.empty((M, M), dtype=object)

    for I in range(M):
        local_space = local_models_data[I]['space']
        local_grid = local_problems[I].grid
        for J in dd_grid.neighbors(I):
            J_space = local_spaces[J]
            coupling_grid = dd_grid.coupling_grid(I, J)
            walker = Walker(coupling_grid)

            def make_fake_dirichlet_coupling_parametric_part(func):
                coupling_form = BilinearForm(coupling_grid)
                dirichlet_coupling_integrand = LocalLaplaceIPDGDirichletCouplingIntegrand(
                    symmetry_factor,
                    GF(local_grid, func, (Dim(d), Dim(d))),
                    intersection_type=CouplingIntersection(dd_grid))
                coupling_form += LocalIntersectionIntegralBilinearForm(
                    dirichlet_coupling_integrand)
                op = MatrixOperator(coupling_grid, source_space=local_space,
                                    range_space=J_space,
                                    sparsity_pattern=local_models_data[I]['sparsity_pattern'])
                op.append(coupling_form, {}, (False, False, False, False, False, True))
                walker.append(op)
                return op

            def make_fake_dirichlet_coupling_nonparametric_part():
                coupling_form = BilinearForm(coupling_grid)
                dirichlet_penalty_integrand = LocalIPDGBoundaryPenaltyIntegrand(
                    penalty_parameter, local_weights[I],
                    intersection_type=CouplingIntersection(dd_grid))
                coupling_form += LocalIntersectionIntegralBilinearForm(
                    dirichlet_penalty_integrand)
                op = MatrixOperator(coupling_grid, source_space=local_space,
                                    range_space=J_space,
                                    sparsity_pattern=local_models_data[I]['sparsity_pattern'])
                op.append(coupling_form, {}, (False, False, False, False, False, True))
                walker.append(op)
                return op

            ops = [make_fake_dirichlet_coupling_parametric_part(func)
                   for func in local_problems[I].diffusion.functions] + \
                [make_fake_dirichlet_coupling_nonparametric_part(), ]
            coeffs = list(local_problems[I].diffusion.coefficients) + [1., ]
            # NOTE: we probably only need the nonparametric part !
            # TODO: check this!!

            walker.walk(False)  # parallel assembly not yet supported

            dirichlet_ops = [DuneXTMatrixOperator(op.matrix, name=f'dirichlet_part_{I}_{J}')
                             for op in ops]
            dirichlet_ops[-1] = dirichlet_ops[-1].with_(name=f'constant_dirichlet_part_{I}_{J}')

            fake_dirichlet_op = LincombOperator(operators=dirichlet_ops, coefficients=coeffs)

            fake_dirichlet_ops[I][J] = fake_dirichlet_op

    fake_dirichlet_op = BlockOperator(fake_dirichlet_ops)

    data = {'macro_grid': macro_grid,
            'dd_grid': dd_grid,
            'macro_boundary_info': macro_boundary_info,
            'local_spaces': local_spaces,
            'coupling_op': coupling_op,
            'lhs_without_coupling': lhs_without_coupling,
            'fake_dirichlet_ops': fake_dirichlet_op}

    if preassemble:
        data['unassembled_m'] = m
        m = preassemble_(m)

    return m, data

def discretize_instationary_ipld3g(analytical_problem, macro_diameter=None,
                                   num_local_refinements=None, penalty_parameter=None,
                                   # domain_discretizer=None, grid_type=None,
                                   # grid=None, boundary_info=None, num_values=None,
                                   time_stepper=None, nt=None,
                                   # preassemble=True
                                   ):
    assert isinstance(analytical_problem, InstationaryProblem)
    assert isinstance(analytical_problem.stationary_part, StationaryProblem)
    # assert grid is None or boundary_info is not None
    # assert boundary_info is None or grid is not None
    # assert grid is None or domain_discretizer is None
    assert (time_stepper is None) != (nt is None)

    p = analytical_problem

    if p.stationary_part.dirichlet_data is not None and 't' in p.stationary_part.dirichlet_data.parameters:
        # we choose both mass and operator to be invertible.
        # this leads to wrong results when the dirichlet values depend on time.
        raise NotImplementedError('Time-dependent Dirichlet values not supported.')

    m, data = discretize_stationary_ipld3g(p.stationary_part, macro_diameter=macro_diameter,
                                           num_local_refinements=num_local_refinements,
                                           penalty_parameter=penalty_parameter,
                                           # domain_discretizer=domain_discretizer,
                                           # grid_type=grid_type, grid=grid, boundary_info=boundary_info,
                                           # preassemble=preassemble
                                           )

    if p.initial_data.parametric:
        assert 0
        I = InterpolationOperator(data['grid'], p.initial_data)
    else:
        # from the standard instationry discretizer
        # I = p.initial_data.evaluate(data['grid'].centers(data['grid'].dim))
        Is = []
        for T in range(data['dd_grid'].num_subdomains):
            local_grid = data['dd_grid'].local_grid(T)
            vector = IstlVector(p.initial_data.evaluate(local_grid.centers(p.initial_data.dim_domain)))
            Is.append(m.solution_space.subspaces[T].make_array([vector]))
        I = m.solution_space.make_array(Is)

    if time_stepper is None:
        if p.stationary_part.diffusion is None:
            time_stepper = ExplicitEulerTimeStepper(nt=nt)
        else:
            time_stepper = ImplicitEulerTimeStepper(nt=nt)

    mass = m.l2_0_product

    m = InstationaryModel(operator=m.operator, rhs=m.rhs, mass=mass, initial_data=I, T=p.T,
                          products=m.products,
                          output_functional=m.output_functional,
                          time_stepper=time_stepper,
                          visualizer=m.visualizer,
                          num_values=None, name=f'{p.name}_CG')

    # if preassemble:
    #     # m has preassembled stationary parts, whose unassembled version we get from data
    #     ua_m = data['unassembled_m']
    #     unassembled_m = m.with_(operator=ua_m.operator, rhs=ua_m.rhs, products=ua_m.products)
    #     data['unassembled_m'] = unassembled_m
    #     m = preassemble_(m)

    return m, data
