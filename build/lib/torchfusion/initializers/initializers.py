from torch.nn.init import *

class Normal(object):
    def __init__(self,mean=0,std=1):
        """

        :param mean:
        :param std:
        """
        self.mean = mean
        self.std = std

    def __call__(self,tensor):

        return normal_(tensor,self.mean,self.std)


class Uniform(object):
    def __init__(self, lower=0, upper=1):
        """

        :param lower:
        :param upper:
        """
        self.lower = lower
        self.upper = upper

    def __call__(self, tensor):

        return uniform_(tensor, self.lower, self.upper)


class Constant(object):
    def __init__(self, value):
        """

        :param value:
        """
        self.value = value

    def __call__(self, tensor):

        return constant_(tensor,self.value)

class Eye(object):
    def __call__(self, tensor):

        return eye_(tensor)

class Dirac(object):
    def __call__(self, tensor):

        return dirac_(tensor)

class Ones(Constant):
    def __init__(self):
        super(Ones,self).__init__(1)


class Zeros(Constant):
    def __init__(self):
        super(Zeros,self).__init__(0)

class Sparse(object):
    def __init__(self, sparsity_ratio,std=0.01):
        """

        :param sparsity_ratio:
        :param std:
        """
        self.sparsity_ratio = sparsity_ratio
        self.std = std

    def __call__(self, tensor):

        return sparse_(tensor,self.sparsity_ratio,self.std)

class Kaiming_Normal(object):
    def __init__(self,neg_slope=0,mode="fan_in",non_linearity="leaky_relu"):
        """

        :param neg_slope:
        :param mode:
        :param non_linearity:
        """
        self.neg_slope = neg_slope
        self.mode = mode
        self.non_linearity = non_linearity

    def __call__(self, tensor):

        return kaiming_normal_(tensor,self.neg_slope,self.mode,self.non_linearity)


class Kaiming_Uniform(object):
    def __init__(self,neg_slope=0,mode="fan_in",non_linearity="leaky_relu"):
        """

        :param neg_slope:
        :param mode:
        :param non_linearity:
        """
        self.neg_slope = neg_slope
        self.mode = mode
        self.non_linearity = non_linearity

    def __call__(self, tensor):

        return kaiming_uniform_(tensor,self.neg_slope,self.mode,self.non_linearity)


class Xavier_Normal(object):
    def __init__(self, gain=1):
        """

        :param gain:
        """
        self.gain = gain
    def __call__(self, tensor):
        return xavier_normal_(tensor,self.gain)

class Xavier_Uniform(object):
    def __init__(self, gain=1):
        """

        :param gain:
        """
        self.gain = gain
    def __call__(self, tensor):
        return xavier_uniform_(tensor,self.gain)

class Orthogonal(object):
    def __init__(self, gain=1):
        """

        :param gain:
        """
        self.gain = gain
    def __call__(self, tensor):
        return orthogonal_(tensor,self.gain)