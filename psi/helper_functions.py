import numpy as np 
import sys

def loading_verbose(string):
	msg = (string)
	sys.stdout.write('\r'+msg)
	sys.stdout.flush()