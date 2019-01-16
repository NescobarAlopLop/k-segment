__author__ = 'Anton'


class Stack(object):
    """
    A very simple implementation for stack data structure.
    """
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def top(self):
        return self.items[-1]

    def size(self):
        return len(self.items)

    def __str__(self):
        return '{}'.format(self.items)

    def __repr__(self):
        return 'Coreset Stack'

    def __len__(self):
        return len(self.items)
