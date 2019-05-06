#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import unittest
from Dividers import Dividers
from Dividers import get_dividers_point_pairs_for_drawing


class DividersTest(unittest.TestCase):

    def setUp(self):
        sub_divs_inner = [Dividers(0, 6), Dividers(7, 9)]
        self.dividers = [Dividers(0, 4, sub_divs_inner), Dividers(5, 9, sub_divs_inner)]

    def test_init(self):
        self.assertEqual(self.dividers[0].begin, 0)
        self.assertEqual(self.dividers[0].end, 4)
        self.assertEqual(self.dividers[1].begin, 5)
        self.assertEqual(self.dividers[1].end, 9)
        self.assertEqual(self.dividers[0].sub_divs[0].begin, 0)
        self.assertEqual(self.dividers[0].sub_divs[0].end, 6)
        self.assertEqual(self.dividers[0].sub_divs[1].begin, 7)
        self.assertEqual(self.dividers[0].sub_divs[1].end, 9)
        self.assertEqual(self.dividers[1].sub_divs[0].begin, 0)
        self.assertEqual(self.dividers[1].sub_divs[0].end, 6)
        self.assertEqual(self.dividers[1].sub_divs[1].begin, 7)
        self.assertEqual(self.dividers[1].sub_divs[1].end, 9)
        self.assertEqual(self.dividers[0].cost, None)
        self.assertEqual(self.dividers[1].cost, None)
        self.assertEqual(self.dividers[0].sub_divs[0].cost, None)
        self.assertEqual(self.dividers[0].sub_divs[1].cost, None)
        self.assertEqual(self.dividers[1].sub_divs[0].cost, None)
        self.assertEqual(self.dividers[1].sub_divs[1].cost, None)

    def test_update_cost(self):
        self.dividers[0].update_cost(0.56793)
        self.assertEqual(self.dividers[0].cost, 0.56793)

    def test_get_dividers_point_pairs_for_drawing(self):
        dividers_points = get_dividers_point_pairs_for_drawing(self.dividers)
        self.assertEqual(len(dividers_points), 16)
        self.assertEqual(dividers_points,
                         [((0, 0), (0, 6)), ((4, 0), (4, 6)), ((0, 0), (4, 0)), ((0, 6), (4, 6)),
                          ((0, 7), (0, 9)), ((4, 7), (4, 9)), ((0, 7), (4, 7)), ((0, 9), (4, 9)),
                          ((5, 0), (5, 6)), ((9, 0), (9, 6)), ((5, 0), (9, 0)), ((5, 6), (9, 6)),
                          ((5, 7), (5, 9)), ((9, 7), (9, 9)), ((5, 7), (9, 7)), ((5, 9), (9, 9))])
