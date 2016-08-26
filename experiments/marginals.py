from vision import *
from vision.alearn import marginals
from vision import visualize, model
from vision.toymaker import *
import multiprocessing
import logging
import pickle

logging.basicConfig(level = logging.INFO)

g = frameiterator("/scratch/vatic/syn-occlusion2")
b = [Box(592, 48, 592 + 103, 48 + 326, 20)]
stop = 22

g = frameiterator("/scratch/virat/frames/VIRAT_S_000302_04_000453_000484")
b = [Box(156, 96, 156 + 50, 96 + 24, 270),
     Box(391, 83, 391 + 48, 83 + 22, 459)]
stop = 500

pool = multiprocessing.Pool(24)

frame, score, path, marginals = marginals.pick(b, g,
                                               last = stop,
                                               pool = pool,
                                               pairwisecost = .01,
                                               dim = (40, 40),
                                               sigma = 1,
                                               erroroverlap = 0.5,
                                               hogbin = 8,
                                               clickradius = 10,
                                               c = 1)

pickle.dump(marginals, open("occlusion.pkl", "w"))

visualize.save(visualize.highlight_paths(g, [path]), lambda x: "tmp/path{0}.jpg".format(x))

print "frame {0} with score {1}".format(frame, score)
