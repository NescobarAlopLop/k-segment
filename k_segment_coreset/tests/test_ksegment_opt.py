import cProfile
import unittest
from io import StringIO
from os import path
from pstats import Stats

import ksegment_opt

cp = cProfile.Profile()

dir_path = path.dirname(path.realpath(__file__))
dataset_dir = '/'.join(dir_path.split('/')[:-2])
dataset_dir = path.join(dataset_dir, 'datasets', 'segmentation')


def dump_profiler_stats(prof, file_name):
    s = StringIO()
    ps = Stats(prof, stream=s).sort_stats('tottime')
    ps.print_stats()

    pre, ext = path.splitext(file_name)
    with open(pre + '.txt', 'w+') as f:
        f.write(s.getvalue())

    prof.dump_stats(pre)


class KSegmentTestOpt(unittest.TestCase):
    def test_basic_demo(self):
        cp.enable()
        fig_path = ksegment_opt.main(path=None, k=4)
        cp.disable()
        dump_profiler_stats(cp, fig_path)
        cp.clear()

    def test_50px_image(self):
        test_file_path = path.join(dataset_dir, 'bar_code_50px.png')
        cp.enable()
        fig_path = ksegment_opt.main(path=test_file_path)
        cp.disable()
        dump_profiler_stats(cp, fig_path)
        cp.clear()

    def test_50px_image_for_a_range_of_ks(self):
        """
        this function will run for a while
        :return: None
        """
        for k in range(3, 10):
            test_file_path = path.join(dataset_dir, 'bar_code_50px.png')
            cp.enable()
            fig_path = ksegment_opt.main(path=test_file_path, k=k)
            cp.disable()
            dump_profiler_stats(cp, fig_path)
            cp.clear()

    def test_full_size_image_for_a_range_of_ks(self):
        """
        this function will run for a couple of hours, maybe days
        :return: None
        """
        for k in range(3, 10):
            test_file_path = path.join(dataset_dir, 'bar_code.png')
            cp.enable()
            fig_path = ksegment_opt.main(path=test_file_path, k=k)
            cp.disable()
            dump_profiler_stats(cp, fig_path)
            cp.clear()
