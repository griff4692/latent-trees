import numpy as np

def render_args(args):
	print ("")
	for arg in vars(args):
		print ("%s=%s" % (arg, getattr(args, arg)))
	print("")

def cudify(args, tensor):
	if args.gpu > -1:
		tensor = tensor.cuda()
	return tensor


def bh(m, go, gi):
	for (i, g) in enumerate(gi):
		isnan = np.any(np.isnan(g.data.numpy()))
		if isnan:
			print("Grad Input for %s is nan" % m.name)
	for (i, g) in enumerate(go):
		isnan = np.any(np.isnan(g.data.numpy()))
		if isnan:
			print("Grad Output for %s is nan" % m.name)
