import numpy as np
import os

from pymor.algorithms.projection import project, ProjectRules
from pymor.operators.symplectic import CanonicalSymplecticFormOperator
from pymor.models.symplectic import BaseQuadraticHamiltonianModel, QuadraticHamiltonianModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.operators.constructions import ConcatenationOperator, IdentityOperator, InverseOperator
from pymor.models.basic import InstationaryModel
from pymor.algorithms.simplify import contract, expand
from pymor.core.base import BasicObject
from numbers import Number


def regular_J(n):
    return NumpyMatrixOperator(
        np.block(
            [
                [np.zeros((n, n)), np.eye(n)], 
                [-np.eye(n), np.zeros((n, n))]
                ]
            )
        )


def project_initial_data_with_op(V_r, W_r, initial_data):
    projection_matrix = W_r.inner(V_r)
    projection_op = NumpyMatrixOperator(projection_matrix)
    inverse_projection_op = InverseOperator(projection_op, 'inverse_projection_op')
    pid = project(initial_data, range_basis=W_r, source_basis=None)
    projected_initial_data = ConcatenationOperator([inverse_projection_op, pid])
    return projected_initial_data

def project_initial_data(V_r, W_r, initial_data):
    return project(initial_data, W_r, None)


class check_PODReductor():
    def __init__(self, fom, V_r, W_r):
        self.fom = fom
        self.V_r = V_r
        self.W_r = W_r

    def reduce(self):
        fom = self.fom
        V_r = self.V_r
        W_r = self.W_r

        projected_initial_data = project_initial_data_with_op(V_r, W_r, fom.initial_data) 

        projected_operator = project(fom.operator, V_r, V_r)     

        # print("check init data range", projected_operator.source, projected_initial_data.range)     

        projected_operators = {
            'mass':              project(fom.mass, V_r, V_r),
            'operator':          projected_operator,
            'rhs':               None,
            'initial_data':      projected_initial_data,
            'products':          None,
            'output_functional': None
        }
        
        rom = InstationaryModel(T=fom.T, time_stepper=fom.time_stepper, num_values=fom.num_values,
                                 error_estimator=None, **projected_operators)
        
        return rom
    

class StructurePreservingPODReductor(BasicObject):
    def __init__(self, fom, V_r, W_r):
        self.fom = fom
        self.V_r = V_r
        self.W_r = W_r

    def reduce(self, dims=None):
        with self.logger.block('Operator projection ...'):
            fom = self.fom
            V_r = self.V_r
            W_r = self.W_r

            projected_initial_data = project_initial_data_with_op(V_r, W_r, fom.initial_data)
            projected_H_op = project(fom.H_op, V_r, V_r)
            projected_J = project(CanonicalSymplecticFormOperator(fom.H_op.source), W_r, W_r)

            projected_operators_quadratic = {
                'H_op':              projected_H_op,
                'h':                 project(fom.h, V_r, None),
                'initial_data':      projected_initial_data,
                'output_functional': None,
                'J':                 projected_J
            }

        with self.logger.block('Building ROM ...'):
            rom = BaseQuadraticHamiltonianModel(
                fom.T,
                time_stepper=fom.time_stepper,
                num_values=fom.num_values,
                name='reduced_' + fom.name,
                **projected_operators_quadratic
            )
        return rom

    def reconstruct(self, u):
        return self.RB[:u.dim//2].lincomb(u.to_numpy())
    

class StructurePreservingPODReductor_changedLHS(BasicObject):
    def __init__(self, fom, V_r, W_r):
        self.fom = fom
        self.V_r = V_r
        self.W_r = W_r

    def reduce(self, dims=None):
        with self.logger.block('Operator projection ...'):
            fom = self.fom
            V_r = self.V_r
            W_r = self.W_r

            projected_initial_data = project_initial_data_with_op(V_r, W_r, fom.initial_data)
            projected_H_op = project(fom.H_op, V_r, V_r)

            projected_J = project(CanonicalSymplecticFormOperator(fom.H_op.source), W_r, W_r)

            projected_operator = ConcatenationOperator([projected_J.H, projected_H_op])

            # numpy way
            n = fom.H_op.source.dim // 2
            # red_dim = len(V_r)
            if n != 0:
                space = fom.H_op.source
                numpy_J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
                vector_array_J = space.from_numpy(numpy_J)

                numpy_projected_J_inside = V_r.inner(vector_array_J.lincomb(V_r.to_numpy()))

                vector_array_J_inverse = -1 * vector_array_J
                numpy_projected_J_inverse_inside = W_r.inner(vector_array_J_inverse.lincomb(W_r.to_numpy()))

                mass = numpy_projected_J_inverse_inside @ numpy_projected_J_inside
                # np.savetxt(os.path.join("mass", f'mass{red_dim}.txt'), mass)
            projected_mass_numpy = NumpyMatrixOperator(mass)

            # pyMOR way
            J_operator = CanonicalSymplecticFormOperator(fom.H_op.source)
            J_inverse = -1 * J_operator
            projected_J_inside = project(J_operator, V_r, V_r)
            projected_J_inverse_inside = project(J_inverse, W_r, W_r)
            projected_mass_pyMOR = ConcatenationOperator([projected_J_inverse_inside, projected_J_inside])


            projected_operators_instationary = {
            'mass':              projected_mass_numpy,
            'operator':          projected_operator,
            'rhs':               project(fom.rhs, V_r, None),
            'initial_data':      projected_initial_data,
            'products':          None,
            'output_functional': None
        }

        with self.logger.block('Building ROM ...'):
            rom = InstationaryModel(
                fom.T, 
                time_stepper=fom.time_stepper,
                num_values=fom.num_values,
                name = 'reduced_' + fom.name,
                **projected_operators_instationary
                )

        return rom

    def reconstruct(self, u):
        return self.RB[:u.dim//2].lincomb(u.to_numpy())
    

class variatonially_consistent_reductor():
    def __init__(self, fom, V_r, W_r):
        self.fom = fom
        self.V_r = V_r
        self.W_r = W_r

    def reduce(self):
        fom = self.fom
        V_r = self.V_r
        W_r = self.W_r


        J_operator = CanonicalSymplecticFormOperator(fom.H_op.source)
        J_operator_transpose = -1 * J_operator
        projected_J = project(J_operator_transpose, V_r, None)
        projected_J_transpose = project(J_operator_transpose, V_r, V_r)
        projected_J_transpose_inverse = InverseOperator(projected_J_transpose)
        P_2 = ConcatenationOperator([projected_J_transpose_inverse, projected_J])
        projected_fom_operator = project(fom.operator, None, V_r)
        projected_H_op = project(fom.H_op, V_r, V_r)
        projected_operator = ConcatenationOperator([P_2, projected_fom_operator])

        projected_initial_data = project_initial_data(V_r, W_r, fom.initial_data) 



        projected_J_2 = project(J_operator_transpose, V_r, V_r)
        projected_mass = ConcatenationOperator([projected_J_transpose_inverse, projected_J_2])
        
        vector_ones = projected_mass.source.from_numpy(np.ones(len(V_r)))
        applied_vector_ones = projected_mass.apply(vector_ones)
        # print("check mass with vector of ones", (projected_mass.apply(vector_ones) - vector_ones).norm2(), applied_vector_ones)

        # print("check init data range", projected_operator.source, projected_initial_data.range)    


        # numpy way
        n = fom.H_op.source.dim // 2
        # red_dim = len(V_r)
        if n != 0:
            space = fom.H_op.source
            numpy_J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
            vector_array_J = space.from_numpy(numpy_J)
            vector_array_J_inverse = -1 * vector_array_J

            # numpy_projected_J_inside = np.linalg.inv(V_r.inner(vector_array_J_inverse.lincomb(V_r.to_numpy())))
            
            numpy_projected_J_inside = V_r.inner(vector_array_J_inverse.lincomb(V_r.to_numpy()))

            mass = np.linalg.inv(numpy_projected_J_inside) @ numpy_projected_J_inside
            # print("check numpy inverse mass", np.linalg.norm(np.identity(mass.shape[0]) - mass))
            # np.savetxt(os.path.join("mass", f'mass{red_dim}.txt'), mass)   
        numpy_projected_mass = NumpyMatrixOperator(mass)

        projected_operators = {
            'mass':              numpy_projected_mass,
            'operator':          projected_operator,
            'rhs':               None,
            'initial_data':      projected_initial_data,
            'products':          None,
            'output_functional': None
        }
        
        rom = InstationaryModel(T=fom.T, time_stepper=fom.time_stepper, num_values=fom.num_values,
                                 error_estimator=None, **projected_operators)
        
        return rom
    


class check_PODReductor_1():
    def __init__(self, fom, V_r, W_r):
        self.fom = fom
        self.V_r = V_r
        self.W_r = W_r

    def reduce(self):
        fom = self.fom
        V_r = self.V_r
        W_r = self.W_r


        J_operator = CanonicalSymplecticFormOperator(fom.H_op.source)
        projected_J = project(J_operator, W_r, W_r)
        J_operator_transpose = -1 * J_operator
        projected_J_transpose = project(J_operator_transpose, V_r, None)
        P_1 = ConcatenationOperator([projected_J, projected_J_transpose])
        projected_fom_operator = project(fom.operator, None, V_r)
        projected_operator = ConcatenationOperator([P_1, projected_fom_operator])

        projected_initial_data = project_initial_data(V_r, W_r, fom.initial_data) 

        # print("check init data range", projected_operator.source, projected_initial_data.range)       

        projected_operators = {
            'mass':              None,
            'operator':          projected_operator,
            'rhs':               None,
            'initial_data':      projected_initial_data,
            'products':          None,
            'output_functional': None
        }
        
        rom = InstationaryModel(T=fom.T, time_stepper=fom.time_stepper, num_values=fom.num_values,
                                 error_estimator=None, **projected_operators)
        
        return rom
    

class check_PODReductor_2():
    def __init__(self, fom, V_r, W_r):
        self.fom = fom
        self.V_r = V_r
        self.W_r = W_r

    def reduce(self):
        fom = self.fom
        V_r = self.V_r
        W_r = self.W_r


        J_operator = CanonicalSymplecticFormOperator(fom.H_op.source)
        projected_J = project(J_operator, W_r, W_r)
        J_operator_transpose = -1 * J_operator
        projected_J_transpose = project(J_operator_transpose, V_r, None)
        P_1 = ConcatenationOperator([projected_J, projected_J_transpose])
        projected_fom_operator = project(fom.operator, None, V_r)
        projected_operator = ConcatenationOperator([P_1, projected_fom_operator])

        projected_initial_data = project_initial_data(V_r, W_r, fom.initial_data) 

        # numpy way
        n = fom.H_op.source.dim // 2
            # red_dim = len(V_r)
        if n != 0:
            space = fom.H_op.source
            numpy_J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
            vector_array_J = space.from_numpy(numpy_J)
            vector_array_J_inverse = -1 * vector_array_J

            numpy_projected_J_inverse_inside = V_r.inner(vector_array_J_inverse.lincomb(V_r.to_numpy()))

            numpy_projected_J_inside = W_r.inner(vector_array_J.lincomb(W_r.to_numpy()))

            mass = numpy_projected_J_inside @ numpy_projected_J_inverse_inside

            # print("check skew-symmetric W^T J W", np.linalg.norm(numpy_projected_J_inside + numpy_projected_J_inside.transpose()))

            # print("check norm of projected mass", np.linalg.norm(mass))
            # np.savetxt(os.path.join("mass", f'mass{red_dim}.txt'), mass)
        projected_mass_numpy = NumpyMatrixOperator(mass)

        # print("check init data range", projected_operator.source, projected_initial_data.range)       

        projected_operators = {
            'mass':              projected_mass_numpy,
            'operator':          projected_operator,
            'rhs':               None,
            'initial_data':      projected_initial_data,
            'products':          None,
            'output_functional': None
        }
        
        rom = InstationaryModel(T=fom.T, time_stepper=fom.time_stepper, num_values=fom.num_values,
                                 error_estimator=None, **projected_operators)
        
        return rom
    



