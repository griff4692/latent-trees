def render_args(args):
	print ("")
	for arg in vars(args):
		print ("%s=%s" % (arg, getattr(args, arg)))
