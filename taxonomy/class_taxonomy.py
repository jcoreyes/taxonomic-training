""" Class for manipulating and viewing class taxonomy """
import pickle
import re
import os
from ete2 import Tree

class ClassTaxonomy():
    def __init__(self, root, adj_file, dataset_dir, collapse=True):
        self.root = root
        self.collapse = collapse
        self.dataset_dir = dataset_dir
        # Adj is dict of parent : list of children
        adj = pickle.load(open(adj_file, 'rb'))
        self.adj = {}
        # Only keep alphanumeric char
        for k, v in adj.items():
            self.adj[''.join(re.findall("[a-zA-Z]+", k))] = [''.join(re.findall("[a-zA-Z]+", x)) for x in v]

        self.tree = self.build_tree()

        self.make_helper_dicts()

    def build_tree(self):
        def build_children(parent):
            """ Construct tree string to load into ETE format ((A, B)C,(D,E)F)G"""
            if parent not in self.adj:
                return parent
            # Collapse internal nodes with only 1 child
            elif self.collapse and len(self.adj[parent]) == 1:
                return build_children(self.adj[parent][0])
            else:
                output = [build_children(x) for x in self.adj[parent]]
                output_str = '(' + ','.join(output) + ')' + parent
                return output_str

        t_str = '(' + build_children(self.root) + ');'

        t = Tree(t_str, format=8)
        return t

    def make_parent_dict(self):
        # Create dict of node: set of all ancestors
        parent_dict = {}
        root = self.tree.get_children()[0]

        def helper(node):
            # Node's parents are it's parent and the parents of it's parent
            parent_dict[node.name] = [node.up.name] + parent_dict[node.up.name]
            for c in node.get_children():
                helper(c)

        parent_dict[root.name] = []
        for c in root.get_children():
            parent_dict[c.name] = root.name
            helper(c)

        return parent_dict

    def make_helper_dicts(self):

        # Leafid to labelidx is dict of leaf label_name : label_idx
        self.leafid_to_labelidx = {}
        classes_file = os.path.join(os.path.expanduser(self.dataset_dir), 'classes.txt')
        for idx, line in enumerate(open(classes_file,'r').readlines()):
            self.leafid_to_labelidx[''.join(re.findall("[a-zA-Z]+", line))] = idx
        self.labelidx_to_leafid = {v: k for k, v in self.leafid_to_labelidx.items()}

        self.leafid_to_parentsid = self.make_parent_dict()

        self.internalid_to_childrenid = {}
        for node in self.tree.get_descendants():
            if node.name not in self.leafid_to_labelidx:
                assert len(node.get_children()) > 0
                self.internalid_to_childrenid[node.name] = [x.name for x in node.get_children()]

        # Dict of leaf label_name : list of (parent node label name, idx of parent node's child that leaf node is under)
        self.leafid_to_internallabels = {}
        for node in self.tree.get_leaves():
            internal_labels = []
            prev = node.name
            # Assumes parents are listed from bottom to top
            for parent in self.leafid_to_parentsid[node.name]:
                internal_labels.append(self.internalid_to_childrenid[parent].index(prev))
                prev = parent
            self.leafid_to_internallabels[node.name] = zip(self.leafid_to_parentsid[node.name], internal_labels)
            self.leafid_to_internallabels[node.name].reverse() # List from broadest to specific
        intersect = set([x.name for x in self.tree.get_leaves()]) - set(self.leafid_to_labelidx)
        assert len(intersect) == 0, intersect

if __name__ == '__main__':
    ctree = ClassTaxonomy('Aves', 'taxonomy_dict.p', '~/NABirds')
    ctree.tree.show()
