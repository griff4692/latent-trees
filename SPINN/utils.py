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
	buf = reversed(range(sent_len))
	stack = []

	for a in act:
		if a == 'shift':
			word_idx = buf.pop()
			new_node = Node(str(word_idx))
			stack.append(new_node)
		elif a == 'reduce':
			noder = stack.pop()
			nodel = stack.pop()

			lLabel = nodel.get_label()
			rLabel = noder.get_label()

			joint = lLabel + "-" + rLabel

			new_node = Node(joint)
			new_node.addKid(nodel)
			new_node.addKid(noder)
		else:
			raise Exception("Should be either shift or reduce!")
	assert len(stack) == 1
	return stack.pop()
