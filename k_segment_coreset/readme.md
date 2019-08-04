# k segment coreset for 2d AKA decision tree coreset

to run opt alg with MSE cost run `ksegment_opt.py` without params

you can also provide either path to image or number of segmetns or both, like so:
`ksegment_opt.py /home/user/image_path.png 4`

be aware running time of optimal alg is exponantiol and may take a very long time.

other interesting files `ksegment.py` and `CoresetKSeg.py`.
`CoresetKSeg.py` contains almost all of the functions needed
for ksegmentation coreset for one dimensional series data. `ksegment.py` has some segmentation helper functions

`utils_seg.py` more helpers

all the files starting with `test_*` are for personal debug use, dont' use to test your code.

`test.py` is actually running some tests, currently not maintained