#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:56:13 2019

@author: danielrubin
"""
import unittest
from unittest import TestCase
from sympy import poly, expand, nan, simplify
from fractions import Fraction
from dcmp import get_sos, get_rational_approximation_one_0_to_1, get_rational_approximation_one


class TestGet_sos(TestCase):

    def test_get_rational_approximation_one_0_to_1_0(self):
        a = 0.5
        max_denom = 2
        num, denom = get_rational_approximation_one_0_to_1(a, max_denom=max_denom)
        self.assertEqual(Fraction(num, denom), Fraction.from_float(a).limit_denominator(max_denom))

    def test_get_rational_approximation_one_0_to_1_1(self):
        a = 0.44
        max_denom = 100
        num, denom = get_rational_approximation_one_0_to_1(a, max_denom=max_denom)
        self.assertEqual(Fraction(num, denom), Fraction.from_float(a).limit_denominator(max_denom))

    def test_get_rational_approximation_one_0_to_1_2(self):
        a = 0.36855050
        max_denom = 10000
        num, denom = get_rational_approximation_one_0_to_1(a, max_denom=max_denom)
        self.assertAlmostEqual(Fraction(num, denom), Fraction.from_float(a).limit_denominator(max_denom))

    def test_get_rational_approximation_one_0_to_1_3(self):
        a = 0.36855050
        max_denom = 10000
        num, denom = get_rational_approximation_one_0_to_1(a, max_denom=max_denom)
        num_, denom_ = get_rational_approximation_one(a, max_denom=max_denom)
        self.assertAlmostEqual(num, num_)
        self.assertAlmostEqual(denom, denom_)

    def test_get_rational_approximation_one_0_to_1_4(self):
        a = 1.5
        max_denom = 3
        num, denom = get_rational_approximation_one(a, max_denom=max_denom)
        self.assertEqual(Fraction(num, denom), Fraction.from_float(a).limit_denominator(max_denom))

    def test_get_rational_approximation_one_0_to_1_5(self):
        a = 2.5
        max_denom = 5
        num, denom = get_rational_approximation_one(a, max_denom=max_denom)
        self.assertEqual(Fraction(num, denom), Fraction.from_float(a).limit_denominator(max_denom))

    def test_get_rational_approximation_one_0_to_1_6(self):
        a = 4.4
        max_denom = 100
        num, denom = get_rational_approximation_one(a, max_denom=max_denom)
        self.assertEqual(Fraction(num, denom), Fraction.from_float(a).limit_denominator(max_denom))

    def test_get_rational_approximation_one_0_to_1_7(self):
        a = 2424242.424242342409
        max_denom = 10000
        num, denom = get_rational_approximation_one(a, max_denom=max_denom)
        self.assertEqual(Fraction(num, denom), Fraction.from_float(a).limit_denominator(max_denom))

    def test_get_sos_odd_0(self):
        polynomial = poly('x')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Polynomial has odd degree. Not a sum of squares.')
        self.assertEqual(expand(sos.as_poly()), nan)

    def test_get_sos_odd_1(self):
        polynomial = poly('abc')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Polynomial has odd degree. Not a sum of squares.')
        self.assertEqual(expand(sos.as_poly()), nan)

    def test_get_sos_odd_2(self):
        polynomial = poly('x**3 + 1')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Polynomial has odd degree. Not a sum of squares.')
        self.assertEqual(sos, nan)

    def test_get_sos_1(self):
        polynomial = poly('x**2 + 2*x + 1')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_am_gm_2(self):
        polynomial = poly('(x**2 + y**2)**2 - 2*x**2*y**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_am_gm_3(self):
        polynomial = poly('(x**2 + y**2 + z**2)**3 - 3*x**2*y**2*z**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_2(self):
        polynomial = poly('(x**2 + y**2 + z**2)**3 - 3*x**2*y**2*z**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_3(self):
        polynomial = poly('x**10 - x**6 - x**4 + 1')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_4(self):
        polynomial = poly('(x + 1)**2*(y-1)**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_5(self):
        polynomial = poly('(x**2 - 1)**2 * (x**2 + 1) * (x**4 + x**2 + 1)')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_le_van_barel_0(self):
        polynomial = poly('1 + x**2 + x**4 + x**6')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_motzkin(self):
        polynomial = poly('1 + x**2*y**2*(x**2 + y**2) - 3*x**2*y**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(simplify(sos).as_expr(), polynomial.as_expr())

    def test_get_sos_motzkin_with_denominator_multiplied_on_left(self):
        polynomial = poly('(1+x**2 + y**2)*(1 + x**2*y**2*(x**2 + y**2) - 3*x**2*y**2)')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def test_get_sos_choi_lam(self):
        polynomial = poly('1 + y**2*z**2 - 4*x*y*z + x**2*z**2 + x**2*y**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(simplify(sos).as_expr(), expand(polynomial).as_expr())

    def test_get_sos_choi_lam_2(self):
        polynomial = poly('y**4 + x**2 - 3*x**2*y**2 + x**4*y**2')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(simplify(sos).as_expr(), polynomial.as_expr())

    def test_get_sos_le_van_barel_1(self):
        polynomial = poly(
            '2*w**2 - 2*z*w + 8*z**2 - 2*y*z + y**2 - 2*y**2*w + y**4 - 4*x*y*z - 4*x**2*z + x**2*y**2 + 2*x**4')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    def mildorf_titu_andreescu(self):
        polynomial = poly('a**4*c**2 + b**4*a**2 + c**4*b**2 - a**3*b**2*c - b**3*c**2*a - c**3*a**2*b')
        status, sos = get_sos(polynomial)
        self.assertEqual(status, 'Exact SOS decomposition found.')
        self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())

    # def mildorf_1(self):
    #     polynomial = poly(
    #         '(a**4*b**2 + b**4*c**2 + c**4*a**2)*(b**4*a**2 + c**4*b**2 + a**4*c**2) - 9*a**4*b**4*c**4')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def mildorf_2(self):
    #     polynomial = poly('a**4*b**2 + b**4*a**2 + 1 - (a**
    #     3*b**2 + b**3*a**2 + ab)')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr()
    #     )
    #
    # def mildorf_5(self):
    #     polynomial = poly(
    #         '(a**2+b**2+c**2+d**2)*b**2*c**2*d**2 +(a**2+b**2+c**2+d**2)*a**2*c**2*d**2 + 4*(a**2+b**2+c**2+d**2)*a**2*b**2*d**2 + 16*(a**2+b**2+c**2+d**2)*a**2*b**2*c**2 - 64*a**2*b**2*c**2*d**2')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def mildorf_8(self):
    #     polynomial = poly('a**6 + b**6 + (1-a**2-b**2)**3 + 6*a**2*b**2*(1-a**2-b**2) - 1/4')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def test_get_sos_le_van_barel_2(self):
    #     polynomial = poly('x**4*y**2 + y**4*z**2 + x**2*z**4 -3*x**2*y**2*z**2 + z**8')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def test_get_sos_imo(self):
    #     # source: https://artofproblemsolving.com/wiki/index.php/2004_USAMO_Problems/Problem_5
    #     polynomial = poly('(a**10 - a**4 + 3)*(b**10 - b**4 + 3)*(c**10 - c**4 + 3) - (a**2 + b**2 + c**2)**3')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
    #
    # def test_get_sos_am_gm_4(self):
    #     polynomial = poly('(x**2 + y**2 + z**2+ t**2)**4 - 4*x**2*y**2*z**2*t**2')
    #     status, sos = get_sos(polynomial)
    #     self.assertEqual(status, 'Exact SOS decomposition found.')
    #     self.assertEqual(expand(sos.as_expr()), expand(polynomial).as_expr())
if __name__ == '__main__':
   unittest.main()