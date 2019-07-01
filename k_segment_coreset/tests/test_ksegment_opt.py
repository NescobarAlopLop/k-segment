import cProfile
import unittest
from io import StringIO
from os import path
from pstats import Stats
from time import time
from matplotlib import pyplot as plt
import numpy as np
import imageio as io
import ksegment_opt
from scipy import ndimage

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
        fig_path = ksegment_opt.main(in_data=None, k=4)
        cp.disable()
        dump_profiler_stats(cp, fig_path)
        cp.clear()

    def test_50px_image(self):
        test_file_path = path.join(dataset_dir, 'bar_code_50px.png')
        cp.enable()
        fig_path = ksegment_opt.main(in_data=test_file_path)
        cp.disable()
        dump_profiler_stats(cp, fig_path)
        cp.clear()

    def test_50px_image_for_a_range_of_ks(self):
        """
        this function will run for a while, around 2 mins per iteration
        :return: None
        """
        for k in range(3, 10):
            test_file_path = path.join(dataset_dir, 'bar_code_50px.png')
            cp.enable()
            fig_path = ksegment_opt.main(in_data=test_file_path, k=k)
            cp.disable()
            dump_profiler_stats(cp, fig_path)
            cp.clear()

    def test_100px_image_for_a_range_of_ks(self):
        """
        this function will run for a while, around one hour per iteration
        :return: None
        """
        for k in range(3, 10):
            test_file_path = path.join(dataset_dir, 'bar_code_100px.png')
            cp.enable()
            fig_path = ksegment_opt.main(in_data=test_file_path, k=k)
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
            fig_path = ksegment_opt.main(in_data=test_file_path, k=k)
            cp.disable()
            dump_profiler_stats(cp, fig_path)
            cp.clear()

    def test_time_complexity(self):

        tmp = []
        for size in range(3, 60):
            # show = True
            for k in range(3, 13, 3):
                test_data = np.random.randint(0, 255, (size, size))
                time_before = time()
                # cp.enable()
                try:
                    fig_path = ksegment_opt.main(in_data=test_data, k=k, show_fig=False)
                    # show = False
                except Exception as e:
                    print(e)
                # cp.disable()
                time_after = time()
                # dump_profiler_stats(cp, fig_path)
                # cp.clear()
                tmp.append((k, size, time_after - time_before, fig_path))

                # if size % 10 == 0:
                #     save_and_plot_stats(tmp, fig_path, k, size)
            save_and_plot_stats(tmp, fig_path, k, size)
        save_and_plot_stats(tmp, fig_path, k, size)

    def test_time_complexity_on_image(self, img_path='/home/george/k-segment-2d/datasets/segmentation/bar_code.png'):
        img = io.imread(img_path, as_gray=True)
        tmp = []

        for zoom in range(3, 100):
            show = True
            for k in range(3, 10):
                test_data = ndimage.zoom(img, zoom / 100.0)
                time_before = time()
                cp.enable()
                try:
                    fig_path = ksegment_opt.main(in_data=test_data, k=k, show_fig=show)
                    show = False
                except Exception as e:
                    print(e)
                cp.disable()
                time_after = time()
                dump_profiler_stats(cp, fig_path)
                cp.clear()
                tmp.append((k, zoom, time_after - time_before, fig_path))

            save_and_plot_stats(tmp, fig_path, k, zoom)
        save_and_plot_stats(tmp, fig_path, k, zoom)


def save_and_plot_stats(in_arr, file_name, k, size):

    dt = np.dtype([('k', np.int), ('size', np.int), ('time', np.float), ('fig_path', np.str)])
    stats = np.array(in_arr, dtype=dt)
    stats_path = file_name.split('_k')[0] + '_k={}_size={}'.format(k, size)

    print("saving to {} with npy extentions".format(stats_path))
    np.save(stats_path, stats)

    plt.clf()
    for k in np.unique(stats['k']):
        current = np.where(stats['k'] == k)
        # print(current)
        plt.plot(stats[current]['size'], stats[current]['time'], label='k={}'.format(k))

    plt.legend()
    plt.show()
    plt.savefig(stats_path)
