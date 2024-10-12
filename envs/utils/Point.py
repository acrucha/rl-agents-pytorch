import math
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        return Point(self.x / scalar, self.y / scalar)

    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def dist_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def angle(self):
        return math.atan2(self.y, self.x)