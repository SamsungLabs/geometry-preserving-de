import torch
import copy

class GridNet(torch.nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = []

        current_branch = None
        for branch in branches:
            branch_class = branch['branch']
            kwargs = copy.deepcopy(branch)
            del kwargs['branch']
            if current_branch is None:
                self.branches.append(branch_class(**kwargs))
            else:
                self.branches.append(branch_class(current_branch, **kwargs))

            current_branch = self.branches[-1]

        self.branches = torch.nn.ModuleList(self.branches)

    def forward(self, input):
        res = input
        
        for branch in self.branches:
            res = branch(res)

        return res