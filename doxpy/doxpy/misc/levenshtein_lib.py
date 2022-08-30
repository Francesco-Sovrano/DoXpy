import Levenshtein

def get_normalized_sintactic_distance(a,b):
	return Levenshtein.distance(a,b)/max(len(a),len(b))

def labels_are_similar(a,b, threshold=0.3):
	return get_normalized_sintactic_distance(a,b) < threshold

def remove_similar_labels(tuple_list, threshold=0.3, key=None):
	if key is None:
		key = lambda x: x[0] if isinstance(x, (list,tuple)) else x
	new_tuple_list = []
	# print('Removing similar labels..')
	for t in tuple_list:
		is_unique = True
		for other_t in new_tuple_list:
			if labels_are_similar(key(t),key(other_t),threshold):
				is_unique = False
				break
		if is_unique:
			new_tuple_list.append(t)
	return new_tuple_list

def labels_are_contained(a,b, threshold=0.3, ordered=False):
	if ordered:
		if len(a) > len(b):
			return False
		min_e, max_e = a,b
	else:
		min_e, max_e = sorted((a,b), key=len)
	return 1 + (Levenshtein.distance(min_e,max_e)-len(max_e))/len(min_e) < threshold

def get_most_similar_label(label,other_label_list):
	distance, most_similar_label = min(map(lambda x: (Levenshtein.distance(label,x),x), other_label_list), key=lambda x:x[0])
	return most_similar_label# if min(1.,distance/len(label)) < 0.2 else label
