#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:34:31 2019

@author: danielrubin
"""

from sympy import expand, Matrix, nan, factor_list, degree_list
from cvxopt import matrix, solvers, spmatrix
from scipy.spatial import ConvexHull
import numpy as np
from scipy.linalg import null_space, orth, eigvalsh
from fractions import Fraction
import numba as nb

DSDP_OPTIONS = {'show_progress': False, 'DSDP_Monitor': 5, 'DSDP_MaxIts': 1000, 'DSDP_GapTolerance': 1e-07,
                'abstol': 1e-07, 'reltol': 1e-06, 'feastol': 1e-07}


@nb.njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@nb.njit
def np_amax(arr, axis):
    return np_apply_along_axis(np.amax, axis, arr)


@nb.njit
def np_amin(arr, axis):
    return np_apply_along_axis(np.amin, axis, arr)


@nb.njit
def get_lattice_pts_in_prism(mat):
    """
    :param mat: matrix matrix of row vectors with integer entries.
    :return: matrix whose rows are all integer lattice points in the smallest rectangular prism
    containing the points of mat.
    """

    size_vector = (np_amax(mat, axis=0) - np_amin(mat, axis=0) + 1).astype(np.int64)
    prod_vec = np.ones(size_vector.shape[0], dtype=np.int64)
    prod = 1
    for i in range(size_vector.shape[0]):
        prod *= size_vector[i]
        prod_vec[i] = prod

    _lattice_pts = np.zeros((prod, size_vector.shape[0]), dtype=np.int64)
    add_vec = np.zeros(size_vector.shape[0], dtype=np.int64)
    for i in range(prod):
        add_vec[0] = i % prod_vec[0]
        for j in range(1, size_vector.shape[0]):
            add_vec[j] = np.int64(i / prod_vec[j - 1]) % size_vector[j]
        _lattice_pts[i] = np_amin(mat, axis=0) + add_vec
    return _lattice_pts


def get_pts_in_cvx_hull(mat, tolerance=1e-03):
    """
    :param mat: matrix whose rows are integer lattice points,
    :param tolerance:
    :return: matrix whose rows are the integer lattice points lying within the convex hull
    of these points to within a given tolerance. This includes the case in which the convex
    hull is less than full dimension.
    """

    nmons = mat.shape[0]
    a_0 = mat - np.repeat([mat[0]], nmons, axis=0)  # Translating so that span goes through origin.
    _null_space = null_space(a_0)
    _integer_pts = get_lattice_pts_in_prism(mat)
    if _null_space.shape[1] == 0:  # In this case, the convex hull has full dimension.
        __integer_pts = _integer_pts
    else:
        _dot_prod = np.abs(
            _integer_pts.dot(_null_space) - np.repeat([mat.dot(_null_space)[0]], _integer_pts.shape[0],
                                                      axis=0))  # Calculate dot product with null vectors
        include = np.all(np.less(_dot_prod, tolerance), axis=1)  # Include only points in same subspace up to tolerance
        __integer_pts = _integer_pts[list(include)]

    _orth = orth(a_0.T)
    if _orth.shape[1] > 1:
        _cvx_hull = ConvexHull(mat.dot(_orth))

        # Now check the points of __integer_pts against the inequalities that define the convex hull
        __cvx_hull = (_cvx_hull.equations[:, :-1].dot(_orth.T.dot(__integer_pts.T))
                      + _cvx_hull.equations[:, -1].reshape((_cvx_hull.equations.shape[0], 1)))
        include = np.all(np.less(__cvx_hull, tolerance), axis=0)
        ___integer_pts = __integer_pts[list(include)]
    else:
        """
        If the linear span of the points of mat is 1- or 0-dimensional, then there is no need to use inequalities given
        the construction of the get_lattice_pts_in_prism function.
        """
        ___integer_pts = __integer_pts
    return ___integer_pts


@nb.njit
def constr_eq_compat(poly_ind, sqroot_monoms):
    """
    :param poly_ind:
    :param sqroot_monoms:
    :return: Boolean value on whether the constraint equations admit a solution.
    The constraint equations are unsatisfiable only if there is a monomial term
    appearing in the polynomial that is not the sum of two points in sqroot_monoms.
    """

    compat = True
    for i in range(poly_ind.shape[0]):
        count = 0
        for j in range(sqroot_monoms.shape[0]):
            for k in range(j, sqroot_monoms.shape[0]):
                if np.all(poly_ind[i] == sqroot_monoms[j] + sqroot_monoms[k]):
                    count += 1
        if count == 0:
            compat = False
    return compat


@nb.njit
def form_constraint_eq_matrices(mat, mat_other):
    """
    :param mat: matrix whose rows are integer vectors.
    :param mat_other: matrix whose rows are integer vectors.
    :return: list of sparse, symmetric matrices, one for each row of mat,
    with size the number of rows of mat_other,
    having a 1 in the entry indexed by (beta,beta') if beta + beta' = alpha,
    and 0 otherwise.
    """

    _mat = []
    for i in range(mat.shape[0]):
        mat_i = np.zeros((mat_other.shape[0], mat_other.shape[0]))
        point_count = 0
        for j in range(mat_other.shape[0]):
            for k in range(j, mat_other.shape[0]):
                if np.all(mat[i] == (mat_other[j] + mat_other[k])):
                    mat_i[j, k] = 1
                    mat_i[k, j] = 1
                    point_count += 1
        if point_count > 0:
            _mat.append(mat_i)
    return _mat


def get_coeffs(poly):
    """
    :param poly: multivariable sympy poly
    :return: vector of coefficients, including zeros for all multi-indices
    in the convex hull of multi-indices appearing in poly.
    Includes case where multi-indices in poly have less than full-dimensional
    convex hull.
    """

    indices = np.array(list(poly.as_poly().as_dict().keys()))
    mat = get_pts_in_cvx_hull(indices)
    mat_other = get_pts_in_cvx_hull(1 / 2 * indices)
    num_nontriv_eq = len(form_constraint_eq_matrices(mat, mat_other))
    coeff_vec = np.zeros(num_nontriv_eq)
    for i in range(num_nontriv_eq):
        if tuple(mat[i]) in poly.as_poly().as_dict().keys():
            coeff_vec[i] = poly.as_poly().as_dict()[tuple(mat[i])]
    return coeff_vec


@nb.njit
def form_sdp_constraint_dense(matrix_list):
    """
    :param matrix_list: list of matrices
    :return: list of matrices of matrix_list each reformatted to the format required by solvers.sdp function.
    """

    num_constr = len(matrix_list)
    row_size = matrix_list[0].shape[0] ** 2
    constr = np.zeros((num_constr, row_size))
    for i in range(num_constr):
        constr[i] = matrix_list[i].reshape((1, row_size))
    return constr


# jit doesn't support sparse matrices (spmatrix) used here
def form_coeffs_constraint_eq_sparse_upper(monoms, sqroot_monoms):
    """
    Forms the coefficients of the constraint equations given matrices monoms, sqroot_monoms
    whose rows correspond to the multi-indices in the convex hull and 1/2 the convex hull
    of the multi-indices of a polynomial.
    Constraint matrices are returned in spmatrix form; only upper triangular elements given.
    :param monoms:
    :param sqroot_monoms:
    :return:
    """

    num = sqroot_monoms.shape[0]
    constraints = []
    for i in range(monoms.shape[0]):
        constraint_i_rows = []
        constraint_i_cols = []
        count_nontriv = 0
        for j in range(num):
            for k in range(j, num):
                if np.all(monoms[i] == (sqroot_monoms[j] + sqroot_monoms[k])):
                    constraint_i_rows.append(j)
                    constraint_i_cols.append(k)
                    count_nontriv += 1
        if count_nontriv:
            constraints.append(spmatrix(1, constraint_i_rows, constraint_i_cols, (num, num)))
    return constraints


@nb.njit
def sym_coeff(x, y):
    if x == y:
        return 1
    else:
        return 2


def get_explicit_form_basis(monoms, sqroot_monoms, poly):
    """
    :param monoms:
    :param sqroot_monoms:
    :param poly: sympy poly
    :return: tuple of symmetric |sqroot_monoms|*|sqroot_monoms| matrices (G_0,G_1,...,G_n),
    n = |sqroot_monoms|*(|sqroot_monoms|+1)/2 - (number of nontrivial constraints),
    where G(y) = G_0 + G_1 y_1 + ... + G_n y_n, y in R^n
    parametrizes the set of Gram matrices for the polynomial.
    """

    dim = sqroot_monoms.shape[0]
    param = int(dim * (dim + 1) / 2)
    constr = form_coeffs_constraint_eq_sparse_upper(monoms, sqroot_monoms)
    gram_mats_sym = []
    coeff = get_coeffs(poly)

    for i in range(param - len(constr) + 1):
        gram_mats_sym.append(np.zeros((dim, dim)))

    num = 1
    for i in range(len(constr)):
        count = len(constr[i].I)
        for j in range(count - 1):
            gram_mats_sym[num + j][constr[i].I[j], constr[i].J[j]] = 1
            gram_mats_sym[num + j][constr[i].I[-1], constr[i].J[-1]] = -sym_coeff(constr[i].I[j],
                                                                                  constr[i].J[j]) / sym_coeff(
                constr[i].I[-1], constr[i].J[-1])
        gram_mats_sym[0][constr[i].I[-1], constr[i].J[-1]] = coeff[i] / sym_coeff(constr[i].I[-1], constr[i].J[-1])
        num += count - 1
    for i in range(len(gram_mats_sym)):
        # make gram_mats_sym[i] symmetric by accessing only upper-triang elts, and copying them onto lower-triang elts:
        gram_mats_sym[i] = np.tril(gram_mats_sym[i].T) + np.triu(gram_mats_sym[i], 1)
    return gram_mats_sym


@nb.njit
def get_rational_approximation_one_0_to_1(x, max_denom):
    """
    :param x: float between 0 and 1
    :param max_denom: max denominator of approximation
    :return: numerator and denominator where denominator < max_denominator such that numerator / denominator is optimal
    rational approximation of x
    """
    a, b = 0, 1
    c, d = 1, 1
    while b <= max_denom and d <= max_denom:
        mediant = float(a + c) / (b + d)
        if x == mediant:
            if b + d <= max_denom:
                return a + c, b + d
            elif d > b:
                return c, d
            else:
                return a, b
        elif x > mediant:
            a, b = a + c, b + d
        else:
            c, d = a + c, b + d
    if b > max_denom:
        return c, d
    else:
        return a, b


@nb.njit
def get_rational_approximation_one(x, max_denom):
    """
    :param x: float
    :param max_denom: max denominator of approximation
    :return: numerator and denominator where denominator < max_denominator such that numerator / denominator is optimal
    rational approximation of x
    """
    x_floor = int(np.floor(x))
    x_frac = x - x_floor
    _num, _denom = get_rational_approximation_one_0_to_1(x_frac, max_denom)
    return _num + _denom * x_floor, _denom


def get_rational_approximation(mat, max_denom):
    """
    :param mat: matrix of floats
    :param max_denom: positive integer
    :return: np.ndarray of Fractions which are the best rational approximations to the entries of mat with denominator
    bounded by max_denom.
    """

    array = np.array(mat)
    rationals = np.zeros_like(mat, dtype=Fraction)
    for (i, j), a in np.ndenumerate(array):
        # rationals[i, j] = Fraction.from_float(a).limit_denominator(max_denom)
        num, denom = get_rational_approximation_one(a, max_denom)
        rationals[i, j] = Fraction(num, denom)
    return rationals


@nb.njit
def get_explicit_rep_objective(sym_matrix_list):
    """
    :param sym_matrix_list:
    :return:
    column objective vector d in the SDP written in explicit form,
    so that the objective is to minimize d^T y.
    The objective is chosen to correspond to the identity matrix in the implicit representation.
    """

    obj_vec = np.zeros((len(sym_matrix_list) - 1, 1))
    for i in range(1, len(sym_matrix_list)):
        obj_vec[i - 1, 0] = np.trace(sym_matrix_list[i])
    return obj_vec


def get_sqroot_monoms(poly):
    """
    :param poly:
    :return: column vector of monomials, the basis of the space of polynomials
    whose square is in the convex hull of the monomials of poly.
    """

    poly_indices = np.array(poly.as_poly().monoms())
    sqroot_monoms = get_pts_in_cvx_hull(1 / 2 * poly_indices)
    monom_vec = Matrix.ones(sqroot_monoms.shape[0], 1)
    for i in range(sqroot_monoms.shape[0]):
        for j in range(sqroot_monoms.shape[1]):
            monom_vec[i, 0] *= poly.as_poly().gens[j] ** sqroot_monoms[i, j]
    return monom_vec


def lu_to_sq_factors(lt, ut, perm, monom_vec):
    """
    :param lt: L in the LU decomposition of a rational PSD Gram matrix
    :param ut: U in the LU decomposition of a rational PSD Gram matrix
    :param perm: list of transpositions returned by LU decomposition
    :param monom_vec: vector of monomials in 1/2*ConvexHull(poly)
    :return: two lists corresponding to the SOS decomposition of poly,
    a list of positive factors, and a list of the polynomial factors to be squared.
    """

    perm_mat = Matrix.eye(lt.shape[0]).permuteFwd(perm)
    perm_vec = perm_mat * monom_vec
    pos_coeffs = []
    for i in range(ut.shape[0]):
        pos_coeffs.append((ut * perm_mat.transpose())[i, i])

    sq_factors = []
    for i in range(lt.shape[0]):
        sq_factors.append(lt[:, i].transpose().dot(perm_vec))
    return pos_coeffs, sq_factors


def sdp_expl_solve(basis_matrices, smallest_eig=0, objective = 'zero', dsdp_solver='dsdp', dsdp_options=DSDP_OPTIONS):
    """
    :param basis_matrices: list of symmetric matrices G_0, G_1, ..., G_n of same size
    :param smallest eig: parameter (default 0) may be set to small positive quantity to force non-degeneracy
    :param objective: string parameter, either 'zero', 'min_trace', or 'max_trace' (default 'zero'), determines 
    the objective in the SDP solver
    :param dsdp_solver: string, default 'dsdp' to specify which solver sdp.solver uses
    :param dsdp_options:
    :return: solver_status, a string, either 'optimal', 'infeasible', or 'unknown', and sol_vec, a vector approximately
    optimizing the SDP problem if solver_status is 'optimal', and nan instead
    """
    
    sym_grams = matrix(form_sdp_constraint_dense(basis_matrices[1:])).T
    if objective == 'zero':
        obj_vec = matrix(np.zeros((len(basis_matrices) - 1, 1)))
    elif objective == 'min_trace':
        obj_vec = matrix(get_explicit_rep_objective(basis_matrices))
    else:
        # Maximize trace in nondegenerate case
        obj_vec = -matrix(get_explicit_rep_objective(basis_matrices))
    
    sol = solvers.sdp(c=obj_vec, Gs=[-sym_grams],
                      hs=[matrix(basis_matrices[0] -smallest_eig*np.eye(basis_matrices[0].shape[0]))],
                      solver = dsdp_solver, options = dsdp_options)
    
    if sol['status'] == 'optimal':
        solver_status = 'Optimal solution found'
        sol_vec = sol['x']
    elif (sol['status'] == 'primal infeasible' or sol['status']=='dual infeasible'):
        solver_status = 'infeasible'
        sol_vec = nan
    else:
        solver_status = 'unknown'
        sol_vec = nan
    
    return solver_status, sol_vec
        

def get_sos_helper2(poly, dsdp_solver='dsdp', dsdp_options=DSDP_OPTIONS, eig_tol=-1e-07, epsilon=1e-07,
                   max_denom_rat_approx=100):
    """
    :param poly: sympy polynomial
    :param dsdp_solver:
    :param dsdp_options:
    :param eig_tol:
    :param epsilon:
    :param max_denom_rat_approx:
    :return: string with status whether poly is a sum of squares of polynomials, and a sympy expression that is
    the SOSRF decomposition of the poly
    """

    poly_indices = np.array(list(poly.as_poly().as_dict().keys()))
    monoms = get_pts_in_cvx_hull(poly_indices)
    sqroot_monoms = get_pts_in_cvx_hull(1 / 2 * poly_indices)
    #num_beta = sqroot_monoms.shape[0]
    sym_mat_list_gram = get_explicit_form_basis(monoms, sqroot_monoms, poly)
    if len(sym_mat_list_gram) > 1:
        solv_status, sol_vec = sdp_expl_solve(sym_mat_list_gram, smallest_eig=10**(-3), objective='max_trace')
        if solv_status=='Optimal solution found':
            gram_mat_q = form_rat_gram_mat(sym_mat_list_gram, sol_vec, maxd=1000)
            psd_status, char_poly = check_PSD_rational(gram_mat_q)
            if psd_status:
                monom_vec = get_sqroot_monoms(poly)
                if check_gram_exact(gram_mat_q, monom_vec, poly)=='exact':
                    sos = form_sos(gram_mat_q,monom_vec)
                    msg = 'Exact SOS decomposition found.'
                    return msg, sos
                else:
                    msg = 'Not an exact Gram matrix.'
                    return msg, nan
            else:
                msg = 'Error. Solution not PSD'
                return msg, nan
        
        else:
            solv_status, sol_vec = sdp_expl_solve(sym_mat_list_gram, smallest_eig=-10**(-7))
            if solv_status=='Optimal solution found':
                gram_mat = form_num_gram_mat(sym_mat_list_gram,sol_vec)
                is_psd, eigs = check_PSD_numerical(gram_mat, eig_tol = -10**(-7))
                if is_psd == 'not PSD':
                    msg = 'No PSD Gram matrix found.'
                    return msg, nan
                
                gram_mat_q = form_rat_gram_mat(sym_mat_list_gram, sol_vec, maxd=100)
                psd_status, char_poly = check_PSD_rational(gram_mat_q)
                if psd_status:
                    monom_vec = get_sqroot_monoms(poly)
                    if check_gram_exact(gram_mat_q, monom_vec, poly)=='exact':
                        sos = form_sos(gram_mat_q,monom_vec)
                        msg = 'Exact SOS decomposition found.'
                        return msg, sos
                    else:
                        msg = 'Not an exact Gram matrix.'
                        return msg, nan
                else:
                    # Try again with larger denominator.
                    gram_mat_q = form_rat_gram_mat(sym_mat_list_gram, sol_vec, maxd=10**11)
                    psd_status, char_poly = check_PSD_rational(gram_mat_q)
                    if psd_status:
                        monom_vec = get_sqroot_monoms(poly)
                        if check_gram_exact(gram_mat_q, monom_vec, poly)=='exact':
                            sos = form_sos(gram_mat_q,monom_vec)
                            msg = 'Exact SOS decomposition found.'
                            return msg, sos
                        else:
                            msg = 'Not an exact Gram matrix.'
                            return msg, nan
                    else:
                        msg = 'Could not find exact PSD Gram matrix.'
                        return msg, nan
                    
            else:
                msg = 'SDP solver could not find solution.'
                return msg, nan
      
    else:
        # Unique Gram matrix. No need for SDP.
        gram_mat_q = get_rational_approximation(sym_mat_list_gram[0],100) # The max denominator here should be changed 
                                                                     # to twice the largest denominator appearing as
                                                                     # a coefficient in poly.
        psd_status, char_poly = check_PSD_rational(gram_mat_q)
        if psd_status:
            monom_vec = get_sqroot_monoms(poly)
            if check_gram_exact(gram_mat_q, monom_vec, poly)=='exact':
                sos = form_sos(gram_mat_q,monom_vec)
                msg = 'Exact SOS decomposition found.'
                return msg, sos
            else:
                msg = 'Not an exact Gram matrix.'
                return msg, nan
        else:
            msg = 'Unique Gram matrix not PSD. Not a sum of squares.'
            return msg, nan
   
    
  
def get_sos_helper(poly, dsdp_solver='dsdp', dsdp_options=DSDP_OPTIONS, eig_tol=-1e-07, epsilon=1e-07,
                   max_denom_rat_approx=100):
    """
    :param poly: sympy polynomial
    :param dsdp_solver:
    :param dsdp_options:
    :param eig_tol:
    :param epsilon:
    :param max_denom_rat_approx:
    :return: string with status whether poly is a sum of squares of polynomials, and a sympy expression that is
    the SOSRF decomposition of the poly
    """

    poly_indices = np.array(list(poly.as_poly().as_dict().keys()))
    monoms = get_pts_in_cvx_hull(poly_indices)
    sqroot_monoms = get_pts_in_cvx_hull(1 / 2 * poly_indices)
    num_beta = sqroot_monoms.shape[0]
    sym_mat_list_gram = get_explicit_form_basis(monoms, sqroot_monoms, poly)
    if len(sym_mat_list_gram) > 1:
        sym_grams = matrix(
            form_sdp_constraint_dense(sym_mat_list_gram[1:])).T  # This could be stored as sparse spmatrix
        obj_vec = matrix(get_explicit_rep_objective(sym_mat_list_gram))
        zero_vec = matrix(np.zeros((len(sym_mat_list_gram) - 1, 1)))
        sol = solvers.sdp(c=obj_vec, Gs=[-sym_grams], hs=[matrix(sym_mat_list_gram[0])], solver=dsdp_solver,
                          options=dsdp_options)
        fail_status = ['primal infeasible', 'dual infeasible']
        if sol['status'] in fail_status:
            msg = 'SDP solver did not find feasible region for Gram matrix.'
            return msg, nan
        elif sol['status'] == 'unknown':
            # print('SDP solver failed to converge or find certificate of infeasibility.')
            # print('Trying again with zero objective.')
            sol = solvers.sdp(c=zero_vec, Gs=[-sym_grams], hs=[matrix(sym_mat_list_gram[0])], solver=dsdp_solver,
                              options=dsdp_options)
            fail_status = ['primal infeasible', 'dual infeasible']
            if sol['status'] in fail_status:
                msg = 'SDP solver did not find feasible region for Gram matrix.'
                return msg, nan
            elif sol['status'] == 'unknown':
                msg = 'SDP solver failed to converge or find certificate of infeasibility. SOS unknown.'
                return msg, nan
        else:
            gram_mat_num_approx = np.zeros((num_beta, num_beta))
            for i in range(len(sym_mat_list_gram) - 1):
                gram_mat_num_approx += sym_mat_list_gram[i + 1] * sol['x'][i]
            # print('Unique Gram matrix: ')
            gram_mat_num_approx = sym_mat_list_gram[0]
            # print(gram_mat_num_approx)
            # print('Eigenvalues: ')
            eigs = eigvalsh(gram_mat_num_approx)
            # print(eigs)

            # Check that numerical Gram matrix is positive semi-definite.
            if eigvalsh(gram_mat_num_approx)[0] < eig_tol:
                sol = solvers.sdp(c=zero_vec, Gs=[-sym_grams],
                                  hs=[matrix(sym_mat_list_gram[0] + epsilon * np.eye(num_beta))],
                                  solver=dsdp_solver, options=dsdp_options)

                fail_status = ['primal infeasible', 'dual infeasible']
                if sol['status'] in fail_status:
                    msg = 'SDP solver did not find feasible region for Gram matrix.'
                    return msg, nan
                elif sol['status'] == 'unknown':
                    msg = 'SDP solver failed to converge or find certificate of infeasibility. SOS unknown.'
                    return msg, nan
                else:
                    gram_mat_num_approx = np.zeros((num_beta, num_beta))
                    for i in range(len(sym_mat_list_gram) - 1):
                        gram_mat_num_approx += sym_mat_list_gram[i + 1] * sol['x'][i]
                    gram_mat_num_approx += sym_mat_list_gram[0]
                    eigs = eigvalsh(gram_mat_num_approx)
                    if eigs[0] < eig_tol:
                        msg = 'Failed to find PSD Gram matrix.'
                        return msg, nan
            rat_approx = get_rational_approximation(sol['x'], max_denom_rat_approx)
            gram_mat_q = np.zeros((num_beta, num_beta), dtype=Fraction)
            for i in range(len(sym_mat_list_gram) - 1):
                gram_mat_q += get_rational_approximation(sym_mat_list_gram[i + 1], max_denom_rat_approx) * rat_approx[i]
            gram_mat_q += get_rational_approximation(sym_mat_list_gram[0], max_denom_rat_approx)
    else:
        # print('Unique Gram matrix: ')
        gram_mat_num_approx = sym_mat_list_gram[0]
        # print(gram_mat_num_approx)
        # print('Eigenvalues: ')
        eigs = eigvalsh(gram_mat_num_approx)
        # print(eigs)

        if eigs[0] < eig_tol:
            msg = 'Unique Gram matrix not PSD.'
            return msg, nan
        gram_mat_q = get_rational_approximation(gram_mat_num_approx, max_denom_rat_approx)
    monom_vec = get_sqroot_monoms(poly)

    first_pos = 0
    eigs = eigvalsh(gram_mat_num_approx)
    while eigs[first_pos] < 0:
        first_pos += 1
        if first_pos == len(eigs):
            msg = 'Error: All eigenvalues of computed Gram matrix are negative.'
            return msg, nan

    min_pos_eig = eigs[first_pos]

    # Check that the rational approximation to the numerical Gram matrix is positive semi-definite.
    char_poly = Matrix(gram_mat_q).charpoly()
    char_coeffs = char_poly.all_coeffs()
    wrong_sign = False
    for i, _c in enumerate(char_coeffs):
        if (-1) ** i * _c < 0:
            wrong_sign = True
    if wrong_sign:
        maxd = int(num_beta ** 2 * min_pos_eig ** (-1))
        rat_approx = get_rational_approximation(sol['x'], maxd)
        gram_mat_q = np.zeros((num_beta, num_beta), dtype=Fraction)
        for i in range(len(sym_mat_list_gram) - 1):
            gram_mat_q += get_rational_approximation(sym_mat_list_gram[i + 1], 10) * rat_approx[i]
        gram_mat_q += get_rational_approximation(sym_mat_list_gram[0], 10)

        char_poly = Matrix(gram_mat_q).charpoly()
        char_coeffs = char_poly.all_coeffs()
        wrong_sign_again = False
        for i in range(len(char_coeffs)):
            if (-1) ** i * char_coeffs[i] < 0:
                wrong_sign_again = True

        if wrong_sign_again:
            msg = 'Could not find an exact Gram matrix.'
            return msg, nan

    # Check that v^T Q v = poly, where v is the monomial vector.
    check_poly = expand((Matrix(monom_vec).transpose() * Matrix(gram_mat_q) * Matrix(monom_vec))[0, 0])
    # print(check_poly)
    if check_poly.as_poly() == poly.as_poly():
        print('Exact rational Gram matrix found.')
    else:
        msg = 'Not an exact Gram matrix.'
        return msg, nan

    # Compute rational LDL^T decomposition, and convert to sum of squares.
    # print(Matrix(gram_mat_q).charpoly())
    lt, ut, perm = Matrix(gram_mat_q).LUdecomposition()
    # print(Matrix(gram_mat_q).LUdecomposition())

    coeffs, factors = lu_to_sq_factors(lt, ut, perm, monom_vec)
    # print(coeffs)
    # print(factors)
    # sos = np.sum([_c * factors[i] ** 2 for i, _c in enumerate(coeffs)]).as_poly()
    sos = np.sum([_c * factors[i] ** 2 for i, _c in enumerate(coeffs)]).as_expr()
    msg = 'Exact SOS decomposition found.'
    return msg, sos

def form_sos(gram_mat_q, monom_vec):
    """
    :param gram_mat_q: a rational symmetric PSD matrix
    :param monom_vec: basis vector of monomials corresponding to gram_mat_q
    :return: sos, an expression consisting of a sum-of-squares decomposition of the polynomial
    with Gram matrix gram_mat_q
    """
    lt, ut, perm = Matrix(gram_mat_q).LUdecomposition()
    # print(Matrix(gram_mat_q).LUdecomposition())

    coeffs, factors = lu_to_sq_factors(lt, ut, perm, monom_vec)
    # print(coeffs)
    # print(factors)
    # sos = np.sum([_c * factors[i] ** 2 for i, _c in enumerate(coeffs)]).as_poly()
    sos = np.sum([_c * factors[i] ** 2 for i, _c in enumerate(coeffs)]).as_expr()
    #msg = 'Exact SOS decomposition found.'
    return sos
  
def check_PSD_rational(sym_rat_mat):
    """
    :param sym_rat_mat: symmetric rational matrix
    :return: not wrong_sign, a boolean expression, True if sym_rat_mat is PSD, False if not, and char_poly,
    a polynomial object, the characteristic polynomial of sym_rat_mat
    """
    char_poly = Matrix(sym_rat_mat).charpoly()
    char_coeffs = char_poly.all_coeffs()
    wrong_sign = False
    for i, _c in enumerate(char_coeffs):
        if (-1) ** i * _c < 0:
            wrong_sign = True
            
    return not wrong_sign, char_poly
  
def check_PSD_numerical(sym_mat, eig_tol=-10**(-7)):
    """
    :param sym_mat: symmetric matrix of floats
    :param eig_tol: float, default -10**(-7)
    :return: string message, either 'PSD' or 'not PSD' according to whether the smallest eigenvalue 
    computed by eigvalsh is greater than eig_tol
    """
    eigs = eigvalsh(sym_mat)
    if eigs[0] < eig_tol:
        status = 'not PSD'
    else:
        status = 'PSD'
    return status, eigs
    
def check_gram_exact(sym_rat_mat, monom_vec, poly):
    """
    :param sym_rat_mat: n*n symmetric matrix of rational numbers
    :param monom_vec: n*1 basis vector of monomials
    :param poly: polynomial
    :return: string, either 'exact' or 'not exact' according to whether monom_vec^T sym_rat_mat monom_vec = poly
    """
    # Check that v^T Q v = poly, where v is the monomial vector.
    check_poly = expand((Matrix(monom_vec).transpose() * Matrix(sym_rat_mat) * Matrix(monom_vec))[0, 0])
    # print(check_poly)
    if check_poly.as_poly() == poly.as_poly():
        status = 'exact'
    else:
        status = 'not exact'
        
    return status
    
def form_rat_gram_mat(basis_matrices, sol_vec_numerical, maxd):
    """
    :param basis_matrices: list of k+1 n*n symmetric matrices
    :param sol_vec_numerical: k*1 vector
    :param maxd: positive integer
    :return: finds best rational approximation rat_approx to sol_vec_numerical for which each entry has denominator
    bounded by maxd, and returns symmetric matrix of rationals basis_matrices[0] + basis_matrices[1]*rat_approx[1]+...
    + basis_matrices[k]*rat_approx[k]
    """
    rat_approx = get_rational_approximation(sol_vec_numerical, maxd)
    gram_mat_q = np.zeros_like(basis_matrices[0], dtype=Fraction)
    for i in range(len(basis_matrices) - 1):
        gram_mat_q += get_rational_approximation(basis_matrices[i + 1], 10) * rat_approx[i]
    gram_mat_q += get_rational_approximation(basis_matrices[0], 10)
        
    return gram_mat_q

def form_num_gram_mat(basis_matrices, sol_vec_numerical):
    """
    :param basis_matrices: list of k+1 n*n symmetric matrices
    :param sol_vec_numerical: k*1 vector
    :return: symmetric matrix  basis_matrices[0] + basis_matrices[1]*sol_vec_numerical[1]+...
    + basis_matrices[k]*sol_vec_numerical[k]
    """
    gram_mat = basis_matrices[0]
    for i in range(len(basis_matrices) - 1):
        gram_mat += basis_matrices[i + 1] * sol_vec_numerical[i]
       
    return gram_mat
  
  
def get_poly_multiplier(poly):
    """
    :param poly: sympy polynomial
    :return: factorisation of poly
    """
    _symbols = [1] + list(poly.free_symbols)
    _mult = np.sum([_s ** 2 for _s in _symbols]).as_poly()
    return _mult


def get_max_even_divisor(polynomial):
    """
    :param polynomial: sympy polynomial
    :return: leading coefficient of poly, max polynomial divisor of poly that's even power, remainder of poly / max_divisor
    """
    _factors = factor_list(polynomial)
    _coeff_leading = _factors[0]
    _factors_non_constant = _factors[1]
    _factors_max_even_divisor = [(_p, 2 * (n // 2)) for (_p, n) in _factors_non_constant]
    _factors_remainder = [(_p, n - 2 * (n // 2)) for (_p, n) in _factors_non_constant]
    _max_even_divisor = np.prod([_p.as_expr() ** n for (_p, n) in _factors_max_even_divisor])
    _remainder = np.prod([_p.as_expr() ** n for (_p, n) in _factors_remainder])
    return _coeff_leading, _max_even_divisor, _remainder


def get_sos(polynomial, max_mult_power=3, dsdp_solver='dsdp', dsdp_options=DSDP_OPTIONS, eig_tol=-1e-07, epsilon=1e-07):
    """
    :param polynomial: sympy polynomial
    :param max_mult_power:
    :param dsdp_solver:
    :param dsdp_options:
    :param eig_tol:
    :param epsilon:
    :return: string with status whether poly is a sum of squares of polynomials, and a sympy expression that is
    the SOSRF decomposition of the poly
    """
    if polynomial == 0:
        _status = 'Zero polynomial.'
        return _status, nan

    # check polynomial is nonconstant
    if np.all([_d == 0 for _d in degree_list(polynomial)]):
        _status = 'Constant polynomial.'
        return _status, nan

    poly_indices = np.array(list(polynomial.as_poly().as_dict().keys()))
    monoms = get_pts_in_cvx_hull(poly_indices)
    num_alpha = monoms.shape[0]

    if not num_alpha:
        _status = 'Error in computing monomial indices.'
        return _status, nan

    degree = polynomial.as_poly().degree()
    if degree % 2:
        _status = 'Polynomial has odd degree. Not a sum of squares.'
        return _status, nan

    coeff_leading, max_even_divisor, remainder = get_max_even_divisor(polynomial)
    if remainder == 1:
        _status = 'Exact SOS decomposition found.'
        sos = coeff_leading * max_even_divisor
    else:
        _mult = get_poly_multiplier(remainder)
        for r in range(max_mult_power):
            print(f'Trying multiplier power: {r}')
            status_, sos_ = get_sos_helper2((_mult ** r * remainder).as_poly())
            if status_ == 'Exact SOS decomposition found.':
                _status = 'Exact SOS decomposition found.'
                sos = (1 / _mult ** r) * coeff_leading * max_even_divisor * sos_.as_expr()
                break
        else:
            _status = 'No exact SOS decomposition found.'
            sos = nan
    return _status, sos