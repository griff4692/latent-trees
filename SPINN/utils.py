from constants import PAD, SHIFT, REDUCE
from math import log
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

def max_height(actions, num_ops):
	batch_size, max_ops = actions.shape
	heights = np.zeros([batch_size, 1])
	min_heights = np.zeros([batch_size, 1])
	max_heights = np.zeros([batch_size, 1])
	for b_id in range(batch_size):
		n_op = num_ops[b_id]
		sent_size = (n_op - 1) // 2
		max_heights[b_id, 0] = sent_size
		min_heights[b_id, 0] = log(max(sent_size, 2), 2)
		stack = []
		for a in actions[b_id]:
			if a == PAD:
				break
			elif a == REDUCE:
				x = stack.pop()
				y = stack.pop()
				stack.append(max(x, y) + 1.0)
			else:
				stack.append(0)
		assert len(stack) == 1
		heights[b_id, 0] = stack[0]

	baseline = (min_heights + max_heights) / 2.0
	return (baseline - heights)/baseline
