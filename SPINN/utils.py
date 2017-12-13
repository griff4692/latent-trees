from zss import simple_distance, Node

def render_args(args):
	print ("")
	for arg in vars(args):
		print ("%s=%s" % (arg, getattr(args, arg)))
	print("")

def cudify(args, tensor):
	if args.gpu > -1:
		tensor = tensor.cuda()
	return tensor

def tree_edit(act1, act2):
	tree1 = make_tree(act1)
	tree2 = make_tree(act2)
	return simple_distance(tree1, tree2)

def make_tree(act):
	sent_len = sum([1 if 'shift' else 0 for a in act])
	num_shifts = 0
	stack = []

	for a in act:
		if a == 'shift':
			new_node = Node(str(num_shifts))
			stack.append(new_node)
			num_shifts += 1
		elif a == 'reduce':
			noder = stack.pop()
			nodel = stack.pop()

			lLabel = Node.get_label(nodel)
			rLabel = Node.get_label(noder)

			joint = lLabel + "-" + rLabel

			new_node = Node(joint)
			new_node.addkid(nodel)
			new_node.addkid(noder)
			stack.append(new_node)
		else:
			raise Exception("Should be either shift or reduce!")
	assert len(stack) == 1
	return stack.pop()

if __name__ == '__main__':
	a1 = ['shift', 'shift', 'shift', 'reduce', 'reduce']
	a2 = ['shift', 'shift', 'reduce', 'shift', 'reduce']

	d = tree_edit(a1, a2)
	print(d)
