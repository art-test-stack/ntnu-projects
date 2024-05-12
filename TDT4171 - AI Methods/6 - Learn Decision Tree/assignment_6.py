import numpy as np
from pathlib import Path
from typing import Tuple



class Node:
    """ Node class used to build the decision tree"""
    def __init__(self):
        self.children = {}
        self.parent = None
        self.attribute = None
        self.value = None

    def classify(self, example):
        if self.value is not None:
            return self.value
        return self.children[example[self.attribute]].classify(example)


def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)"""
    labels = examples[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count

    return value


def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """
    This function should compute the importance of each attribute and choose the one with highest importance,
    A ← argmax a ∈ attributes IMPORTANCE (a, examples) (Figure 19.5)

    Parameters:
        attributes (np.ndarray): The set of attributes from which the attribute with highest importance is to be chosen
        examples (np.ndarray): The set of examples to calculate attribute importance from
        measure (str): Measure is either "random" for calculating random importance, or "information_gain" for
                        caulculating importance from information gain (see Section 19.3.3. on page 679 in the book)

    Returns:
        (int): The index of the attribute chosen as the test

    """
    if measure == 'information_gain':
        def B(q):
            if q == 0 or q == 1: return 0
            return - (q * np.log2(q) + (1 - q) * np.log2(1 - q))
        
        train_exs = examples.T[:len(examples.T) - 1].T
        classes = examples.T[len(examples.T) - 1].T

        available_values = { a: [] for a in attributes }
        for tr_eg in train_exs:
            for a in attributes:
                if tr_eg[a] not in available_values[a]:
                    available_values[a].append(tr_eg[a])
        available_values = { a: np.sort(np.array(available_values[a])) for a in attributes }
        
        classes_available = { 'yes': 2, 'no': 1 }

        entropy = B(np.count_nonzero(classes == 2) / len(classes))

        remainder = { a: 0 for a in attributes }
        for a in attributes:
            for v in available_values[a]:
                n = np.count_nonzero(train_exs.T[a] == v)
                q = np.count_nonzero((train_exs.T[a] == v) * (classes.T  == classes_available['yes']))
                remainder[a] += n / len(train_exs) * B(q / n)
        gain = np.array([ entropy - remainder[a] for a in attributes ])
        A = attributes[np.argmax(gain)]

    elif measure == 'random':
        import random
        A = random.choice(attributes)
    else:
        assert False, f"Measure {measure} not implemented yet 😞"
    return A


def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str):
    """
    This is the decision tree learning algorithm. The pseudocode for the algorithm can be
    found in Figure 19.5 on Page 678 in the book.

    Parameters:
        examples (np.ndarray): The set data examples to consider at the current node
        attributes (np.ndarray): The set of attributes that can be chosen as the test at the current node
        parent_examples (np.ndarray): The set of examples that were used in constructing the current node’s parent.
                                        If at the root of the tree, parent_examples = None
        parent (Node): The parent node of the current node. If at the root of the tree, parent = None
        branch_value (int): The attribute value corresponding to reaching the current node from its parent.
                        If at the root of the tree, branch_value = None
        measure (str): The measure to use for the Importance-function. measure is either "random" or "information_gain"

    Returns:
        (Node): The subtree with the current node as its root
    """
    node = Node()
    if parent is not None:
        parent.children[branch_value] = node
        node.parent = parent
    
    if examples.size == 0: 
        node.value = plurality_value(parent_examples)
    
    elif (np.all(examples.T[len(examples.T) - 1, :] == examples.T[len(examples.T) - 1, 0])):
        node.value = examples.T[len(examples.T) - 1, 0]
    
    elif attributes.size == 0: 
        node.value = plurality_value(parent_examples)
    
    else: 
        A = importance(attributes, examples, measure)
        node.attribute = A
        max_v_for_a = 0
        for tr_eg in examples.T[:len(examples.T) - 1].T:
            if tr_eg[A] > max_v_for_a: max_v_for_a = tr_eg[A] 

        # NOTE: not written in the book, but if we take only the values from train, 
        #       then it could appear an error if, for eg, a branch has been pruned
        #       from previous call, then no value for training => error. In that 
        #       case the result is always False then.
        values_for_A = range(1, max_v_for_a + 1)

        for v in values_for_A:
            sub_examples = np.array([ eg for eg in examples if eg[A] == v])
            subtree = learn_decision_tree(
                examples=sub_examples, 
                attributes=np.delete(attributes, np.where(attributes==A)),
                parent_examples=examples,
                parent=node, 
                branch_value=v, 
                measure=measure
            )

    return node


def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    i = 0
    index = []
    for example in examples:
        pred = tree.classify(example[:-1])
        is_correct = pred == example[-1]
        correct += pred == example[-1]
        if not is_correct: 
            index.append(i)
        i += 1
    
    return correct / examples.shape[0]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test


if __name__ == '__main__':

    train, test = load_data()

    # information_gain or random
    measure = "information_gain"

    tree = learn_decision_tree(examples=train,
                    attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                    parent_examples=None,
                    parent=None,
                    branch_value=None,
                    measure=measure)

    print(f"Training Accuracy {accuracy(tree, train)}")
    print(f"Test Accuracy {accuracy(tree, test)}")
