import json
from more_itertools import unique_everseen
from doxpy.misc.utils import flatten

class AdjacencyRow:
	get_in_tuple = lambda x: (x[1],x[0])
	get_out_tuple = lambda x: (x[1],x[2])

	def __init__(self, in_=None, out_=None):
		self._in_ = []
		self._out_ = []

	@property
	def in_generator(self):
		return map(AdjacencyRow.get_in_tuple, self._in_)
	
	@property
	def out_generator(self):
		return map(AdjacencyRow.get_out_tuple, self._out_)

	def get_generator(self, direction):
		return self.in_generator if direction == 'in' else self.out_generator

	def add_triplet(self, direction, triplet):
		assert direction in ('in','out')
		if direction == 'in':
			self._in_.append(triplet)
		else:
			self._out_.append(triplet)

	def remove_duplicates(self):
		self._in_ = list(unique_everseen(self._in_))
		self._out_ = list(unique_everseen(self._out_))

	def sort(self):
		self._in_ = sorted(self._in_,key=str)
		self._out_ = sorted(self._out_,key=str)

class AdjacencyList:
	
	def __init__(self, graph, equivalence_relation_set=None, is_sorted=False): # Build the adjacency matrix, for both incoming and outcoming edges
		# self.graph = graph
		self.equivalence_matrix = {}
		self.adjacency_list = {}
		for c in unique_everseen(flatten(map(lambda x: (x[0],x[-1]), graph))): # for every subject and object in the graph
			self.adjacency_list[c] = AdjacencyRow()
		for triplet in graph:
			s,_,o = triplet
			self.adjacency_list[s].add_triplet('out',triplet)
			self.adjacency_list[o].add_triplet('in',triplet)
		if equivalence_relation_set:
			for s,p,o in filter(lambda x: x[1] in equivalence_relation_set, graph):
				if s not in self.equivalence_matrix:
					self.equivalence_matrix[s] = set()
				if o not in self.equivalence_matrix:
					self.equivalence_matrix[o] = set()
				self.equivalence_matrix[s].add(o)
				for e in self.equivalence_matrix[s]:
					if e == o:
						continue
					self.equivalence_matrix[e].add(o)
				self.equivalence_matrix[o].add(s)
				for e in self.equivalence_matrix[o]:
					if e == s:
						continue
					self.equivalence_matrix[e].add(s)
			# print(json.dumps(dict(map(lambda x:(x[0],list(x[1])), self.equivalence_matrix.items())), indent=4))
			for triplet in graph:
				s,_,o = triplet
				for e in self.equivalence_matrix.get(s,[]):
					self.adjacency_list[e].add_triplet('out',triplet)
				for e in self.equivalence_matrix.get(o,[]):
					self.adjacency_list[e].add_triplet('in',triplet)
		# tuplify and remove duplicates
		for arow in self.adjacency_list.values():
			arow.remove_duplicates()
		# print(json.dumps(self.adjacency_list['my:cem'], indent=4))
		if is_sorted:
			for arow in self.adjacency_list.values():
				arow.sort()

	def get_incoming_edges_matrix(self, concept):
		adjacency_list = self.adjacency_list.get(concept,None)
		return tuple(adjacency_list.in_generator) if adjacency_list else []

	def get_outcoming_edges_matrix(self, concept):
		adjacency_list = self.adjacency_list.get(concept,None)
		return tuple(adjacency_list.out_generator) if adjacency_list else []

	def get_edges_between_nodes(self, a, b):
		in_edge_iter = self.get_incoming_edges_matrix(a)
		in_edge_iter = filter(lambda x: x[-1]==b, in_edge_iter)
		edges_between_nodes = set(map(lambda x: (b,x[0],a), in_edge_iter))
		out_edge_iter = self.get_outcoming_edges_matrix(a)
		out_edge_iter = filter(lambda x: x[-1]==b, out_edge_iter)
		edges_between_nodes.update(map(lambda x: (a,x[0],b), out_edge_iter))
		return edges_between_nodes

	def get_equivalent_concepts(self, concept):
		return set(self.equivalence_matrix.get(concept,[]))

	def get_nodes(self):
		return self.adjacency_list.keys()

	def get_predicate_chain(self, concept_set, direction_set, predicate_filter_fn=None, depth=None, already_explored_concepts_set=None): 
		# This function returns the related concepts of a given concept set for a given type of relations (e.g. if the relation is rdfs:subclassof, then it returns the super- and/or sub-classes), exploting an adjacency matrix
		if depth:
			depth -= 1
		if not already_explored_concepts_set:
			already_explored_concepts_set = set()
		joint_set = set()
		already_explored_concepts_set |= concept_set
		for c in concept_set:
			for direction in direction_set:
				adjacency_list = self.adjacency_list.get(c,None)
				if adjacency_list:
					adjacency_iter = filter(lambda x: x[-1] not in already_explored_concepts_set, adjacency_list.get_generator(direction))
					if predicate_filter_fn:
						adjacency_iter = filter(lambda x: predicate_filter_fn(x[0]), adjacency_iter)
					joint_set |= set(map(lambda y: y[-1], adjacency_iter))
		if len(joint_set) == 0:
			return set(concept_set)
		elif depth and depth <= 0:
			return joint_set.union(concept_set)
		return concept_set.union(self.get_predicate_chain(
			joint_set, 
			direction_set,
			predicate_filter_fn=predicate_filter_fn, 
			depth=depth, 
			already_explored_concepts_set=already_explored_concepts_set,
		))

	def get_paths_to_target(self, source, target_set, direction_set, predicate_filter_fn=None, already_explored_concepts_set=None, path_set=None): 
		path = []
		adjacency_list = self.adjacency_list.get(source,None)
		if not adjacency_list:
			return path

		if not already_explored_concepts_set:
			already_explored_concepts_set = set()
		already_explored_concepts_set.add(source)
		target_set.discard(source)

		adjacency_iter = (adjacency for direction in direction_set for adjacency in adjacency_list.get_generator(direction))
		adjacency_iter = filter(lambda x: x[-1] not in already_explored_concepts_set, adjacency_iter)
		if predicate_filter_fn:
			adjacency_iter = filter(lambda x: predicate_filter_fn(x[0]), adjacency_iter)
		adjacency_list = tuple(adjacency_iter)
		if path_set is None:
			path_set = set()
		already_explored_concepts_set |= set(map(lambda x:x[-1], adjacency_list)) - target_set - path_set
		for predicate,target in adjacency_list:
			if target in target_set or target in path_set:
				path.append((source,predicate,target))
				path_set.add(target)
				already_explored_concepts_set.discard(target)
				continue
			paths_to_target = self.get_paths_to_target(target, target_set, direction_set, predicate_filter_fn, already_explored_concepts_set, path_set)
			if paths_to_target:
				paths_to_target_set = set(map(lambda x:x[-1], paths_to_target))
				path.append((source,predicate,target))
				path += paths_to_target
				path_set.add(target)
				path_set |= paths_to_target_set
				already_explored_concepts_set.discard(target)
				already_explored_concepts_set -= paths_to_target_set
		return path

	def get_predicate_dict(self, predicate, manipulation_fn=lambda x: x): # Build labels dict
		predicate_dict = {}
		for s in self.get_nodes():
			for _,o in filter(lambda x: x[0]==predicate, self.get_outcoming_edges_matrix(s)):
				plist = predicate_dict.get(s, None)
				if not plist:
					plist = predicate_dict[s] = []
				plist.append(manipulation_fn(o))
		for k,v in predicate_dict.items():
			predicate_dict[k] = tuple(sorted(v))
		return predicate_dict

	# Tarjan's algorithm (single DFS) for finding strongly connected components in a given directed graph
	def SCC(self): # Complexity : O(V+E) 
		'''A recursive function that finds and prints strongly connected 
		components using DFS traversal 
		u --> The vertex to be visited next 
		disc[] --> Stores discovery times of visited vertices 
		low[] -- >> earliest visited vertex (the vertex with minimum 
					discovery time) that can be reached from subtree 
					rooted with current vertex 
		 st -- >> To store all the connected ancestors (could be part 
			   of SCC) 
		 stackMember[] --> bit/index array for faster check whether 
					  a node is in stack 
		'''
		def helper(clique_list, u, low, disc, stackMember, st, Time=0): 
			# Initialize discovery time and low value 
			disc[u] = Time 
			low[u] = Time 
			Time += 1
			stackMember[u] = True
			st.append(u) 

			# Go through all vertices adjacent to this 
			for _,v in self.adjacency_list[u].in_generator: 
				  
				# If v is not visited yet, then recur for it 
				if disc[v] == -1: 
					Time = helper(clique_list, v, low, disc, stackMember, st, Time) 

					# Check if the subtree rooted with v has a connection to 
					# one of the ancestors of u 
					# Case 1 (per above discussion on Disc and Low value) 
					low[u] = min(low[u], low[v]) 
							  
				elif stackMember[v] == True:  

					'''Update low value of 'u' only if 'v' is still in stack 
					(i.e. it's a back edge, not cross edge). 
					Case 2 (per above discussion on Disc and Low value) '''
					low[u] = min(low[u], disc[v]) 

			# head node found, pop the stack and print an SCC 
			w = -1 #To store stack extracted vertices 
			if low[u] == disc[u]:
				clique = [] 
				while w != u: 
					w = st.pop() 
					clique.append(w)
					stackMember[w] = False
				clique_list.append(clique)
			return Time  
		# Mark all the vertices as not visited  
		# and Initialize parent and visited,  
		# and ap(articulation point) arrays 
		disc = {k:-1 for k in self.adjacency_list.keys()}
		low = {k:-1 for k in self.adjacency_list.keys()}
		stackMember = {k:False for k in self.adjacency_list.keys()}
		st =[] 
		  

		# Call the recursive helper function  
		# to find articulation points 
		# in DFS tree rooted with vertex 'i' 
		clique_list = []
		Time = 0
		for i in self.adjacency_list.keys(): 
			if disc[i] == -1: 
				Time = helper(clique_list, i, low, disc, stackMember, st, Time)
		return clique_list
