from dolfin import *
from fenics import *
from mshr import *
import matplotlib.pyplot as plt


tol=1E-12

space = Rectangle(Point(0,0), Point(1,1))
cylinder = Circle(Point(0.5, 0.55), 0.01)
domain = space - cylinder
mesh = generate_mesh(domain, 100 )

Q = FunctionSpace(mesh, 'Lagrange', 1)

def boundary(x):
    return abs(x[0]) < tol or abs(x[0]) > 1 - tol

wall = 'near(x[1], 0)'
pillar = 'on_boundary && x[0]>0.49 && x[0]<0.51 && x[1]>0.54 && x[1]<0.56'

bc_wall = DirichletBC(Q, Constant(0), wall)
bc_pillar = DirichletBC(Q, Constant(40000), pillar)
bc1 = [bc_wall, bc_pillar]

g  = TrialFunction(Q)
v  = TestFunction(Q)

k=Constant(0)

aL = inner(grad(g), grad(v))*dx
LL = k*v*dx

g=Function(Q)

solve(aL==LL, g, bc1)

Laplace=plot(g, title='show g')
plt.colorbar(Laplace)
plt.show()



class Column(SubDomain):
    def inside(self, x, on_boundary):
        
        return (between(x[0], (0.49, 0.51 )) and between(x[1], (0.54, 0.56)))


column = Column()

boundariesC = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundariesC.set_all(0)
column.mark(boundariesC, 1)

ds = ds(subdomain_data=boundariesC) 

ho0=TrialFunction(Q)
s=TestFunction(Q)
t = TestFunction(Q)

ho0= Function(Q)

F0 = dot((-1*(ho0**2)/0.001),s)* ds(1) + dot((40000-g),t)*ds(1)

ho0J=derivative(F0,ho0)



Ho0problem = NonlinearVariationalProblem(F0, ho0, None , ho0J)

Ho0solver = NonlinearVariationalSolver(Ho0problem)

Ho0prm = Ho0solver.parameters

Ho0solver.solve()

ho0plot=plot(ho0, title='show Ho0')
plt.colorbar(ho0plot)
plt.show()

print ('rho0 calculated')