import Assignment as a
import copy

class CorrectionCSP(a.CSP):

    def backtrack(self, assignment):
        # self.backtrack_calls += 1
        if all([len(assignment[key]) == 1 for key in assignment]):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in assignment[var]:
            assignment_copy = copy.deepcopy(assignment)
            assignment_copy[var] = [value]
            if self.inference(assignment_copy, self.get_all_neighboring_arcs(var)):
                result = self.backtrack(assignment_copy)
                if result:
                    return result
        # self.backtrack_false_returns += 1
        return False

    def select_unassigned_variable(self, assignment):
        # print('assignment', assignment)
        for key in assignment:
            if len(assignment[key]) > 1:
                return key
        # return [key for key in assignment if len(assignment[key]) > 1][0]

    def inference(self, assignment, queue):
        print('queue', queue)
        while queue:
            i, j = queue.pop(0)
            print('i, j ->', i, j)
            if self.revise(assignment, i, j):
                if len(assignment[i]) == 0:
                    return False
                for k in self.get_all_neighboring_arcs(i):
                    if k[0] != j:
                        queue.append(k)
        return True

    def revise(self, assignment, i, j):
        revised = False
        if len(assignment[j]) == 1:        
            for x in assignment[i]:
                if x == assignment[j][0]:
                    assignment[i].remove(x)
                    revised = True
        return revised


def create_map_coloring_csp_c():
    """Instantiate a CSP representing the map coloring problem from the
    textbook. This can be useful for testing your CSP solver as you
    develop your code.
    """
    csp = CorrectionCSP()
    states = ['WA', 'NT', 'Q', 'NSW', 'V', 'SA', 'T']
    edges = {'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
             'NT': ['WA', 'Q'], 'NSW': ['Q', 'V']}
    colors = ['red', 'green', 'blue']
    for state in states:
        csp.add_variable(state, colors)
    for state, other_states in edges.items():
        for other_state in other_states:
            csp.add_constraint_one_way(state, other_state, lambda i, j: i != j)
            csp.add_constraint_one_way(other_state, state, lambda i, j: i != j)
    return csp


def create_sudoku_csp_c(filename: str) -> CorrectionCSP:
    """Instantiate a CSP representing the Sudoku board found in the text
    file named 'filename' in the current directory.

    Parameters
    ----------
    filename : str
        Filename of the Sudoku board to solve

    Returns
    -------
    CSP
        A CSP instance
    """
    csp = CorrectionCSP()
    board = list(map(lambda x: x.strip(), open(filename, 'r')))

    for row in range(9):
        for col in range(9):
            if board[row][col] == '0':
                csp.add_variable('%d-%d' % (row, col), list(map(str,
                                                                range(1, 10))))
            else:
                csp.add_variable('%d-%d' % (row, col), [board[row][col]])

    for row in range(9):
        csp.add_all_different_constraint(['%d-%d' % (row, col)
                                          for col in range(9)])
    for col in range(9):
        csp.add_all_different_constraint(['%d-%d' % (row, col)
                                         for row in range(9)])
    for box_row in range(3):
        for box_col in range(3):
            cells = []
            for row in range(box_row * 3, (box_row + 1) * 3):
                for col in range(box_col * 3, (box_col + 1) * 3):
                    cells.append('%d-%d' % (row, col))
            csp.add_all_different_constraint(cells)

    return csp