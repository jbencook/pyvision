from vision import *
from vision.alearn import marginals
from vision import visualize, model
from vision.toymaker import *
import os
import multiprocessing
import logging
import random
import pylab
import pickle

logging.basicConfig(level = logging.INFO)

g = Geppetto((720,480))

b = Rectangle((400, 100), color="white")
b.linear((400, 800), 20)
g.add(b)

pool = multiprocessing.Pool(24)
frame, score, path, m = marginals.pick([b[0], b[-1]], g, pool = pool,
                                    pairwisecost = .001,
                                    sigma = .1,
                                    erroroverlap = 0.5)

print "frame {0} with score {1}".format(frame, score)
