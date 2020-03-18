from sympy import *


# Field interpolations are tri-linear (linear in x times linear in y
# times linear in z). This amounts to the 3-D generalisation of "area
# weighting". A modification of the simple linear interpolation formula
#           f(i+dx) = f(i) + dx * (f(i+1)-f(i))
# is needed since fields are recorded at half-integer ::locations in certain
# dimensions: see comments and illustration with the Maxwell part of this
# code. One then has first to interpolate from "midpoints" to "gridpoints"
# by averaging neighbors. Then one proceeds with normal interpolation.
# Combining these two steps leads to:
#   f at location i+dx  = half of f(i)+f(i-1) + dx*(f(i+1)-f(i-1))
# where now f(i) means f at location i+1/2. The halving is absorbed
# in the final scaling.
#   E-component interpolations:
#
#
# final interpolation formulas for ex and bx:
#
# f = ex(i,j,k) + ex(i-1,j,k) +    dx*(ex(i+1,j  ,k   ) - ex(i-1,j  ,k  ));
# f+=                                      dy*(ex(i  ,j+1,k   ) + ex(i-1,j+1,k  )+
#                                          dx*(ex(i+1,j+1,k   ) - ex(i-1,j+1,k  ))-f);
# g = ex(i,j,k+1)+ex(i-1,j,k+iz)+  dx*(ex(i+1,j  ,k+iz) - ex(i-1,j  ,k+iz));
# g+=                                      dy*(ex(i  ,j+1,k+iz) + ex(i-1,j+1,k+iz)
#                                      +   dx*(ex(i+1,j+1,k+iz) - ex(i-1,j+1,k+iz))-g);
# ex[n] = (f+dz*(g-f))*0.5;
# 
# 
# 
# f = bx(i,j-1,k)  +bx(i,j-1,k-iz )   +dz*(bx(i,j-1,k+iz)   - bx(i,j-1,k-iz));
# f = bx(i,j,k)    +bx(i,j,k-iz)      +dz*(bx(i,j,k+iz)     - bx(i,j,k-iz))+f+dy 
#   *(bx(i,j+1,k)  +bx(i,j+1,k-iz)    +dz*(bx(i,j+1,k+iz)   - bx(i,j+1,k-iz))-f);
# g = bx(i+1,j-1,k)+bx(i+1,j-1,k-iz)  +dz*(bx(i+1,j-1,k+iz) - bx(i+1,j-1,k-iz));
# g = bx(i+1,j,k)  +bx(i+1,j,k-iz)    +dz*(bx(i+1,j,k+iz)   - bx(i+1,j,k-iz))
#                              +g     +dy*(bx(i+1,j+1,k)    + bx(i+1,j+1,k-iz)
#                                     +dz*(bx(i+1,j+1,k+iz) - bx(i+1,j+1,k-iz))-g);
# bx[n]=(f+dx*(g-f))*(.25);


i, j, k = symbols("i j k", integer=True)
iz = symbols("iz", integer=True)
d, dx, dy, dz = symbols('d dx dy dz')


#ex = Function('ex')
#exi = ex(i-d/2) + d*(ex(i+1-d/2) - ex(i-d/2))

#--------------------------------------------------

ex = Function('ex')

f = ex(i,j,k) + ex(i-1,j,k) +    dx*(ex(i+1,j  ,k   ) - ex(i-1,j  ,k  ))
f+=                                      dy*(ex(i  ,j+1,k   ) + ex(i-1,j+1,k  )+
                                         dx*(ex(i+1,j+1,k   ) - ex(i-1,j+1,k  ))-f)
g = ex(i,j,k+1)+ex(i-1,j,k+iz)+  dx*(ex(i+1,j  ,k+iz) - ex(i-1,j  ,k+iz))
g+=                                      dy*(ex(i  ,j+1,k+iz) + ex(i-1,j+1,k+iz)
                                     +   dx*(ex(i+1,j+1,k+iz) - ex(i-1,j+1,k+iz))-g)
Ex_orig = (f+dz*(g-f))*0.5

print("Original ex:")
print(simplify(Ex_orig.subs([(i,0),(j,0),(k,0),(iz,0), (dz,0)])))


#--------------------------------------------------

d = symbols('Delta')

# interpolation coefficients
#C = Array([
#    1/2*(d*d - d + 1/4),
#    3/4 - d*d,
#    1/2*(d*d + d + 1/4)
#    ])

# capture 1D linear interp
C = Array([
    (d-0.5-1)*0.5,
    0.5,
    (d-0.5)*0.5
    ])

#print(C)

C = Array([
    (d-1)*0.5,
    1.0,
    (d)*0.5
    ])

f = Function('ex')
#Cdx = C.subs(d, dx -1 +0.5)
#Cpx = C.subs(d, dx    +0.5)
#Cpy = C.subs(d, dy    +0.5)
#Cdy = C.subs(d, dy -1 +0.5)

Cdx = C.subs(d, dx -1)
Cpx = C.subs(d, dx   )
Cpy = C.subs(d, dy   )
Cdy = C.subs(d, dy -1)

#another type of staggered grid
#Cdx = C.subs(d, dx-0.5)
#Cpx = C.subs(d, dx+0.5)
#
#Cdy = C.subs(d, dy-0.5)
#Cpy = C.subs(d, dy+0.5)

#print(Cdx)
#print(Cpy)
print(" ")
print(" ")
print("oneD:")
oned = Sum(Cpx[i+1]*f(i), (i, -1, 1)).doit()
print(simplify(oned))
#   f at location i+dx  = half of f(i)+f(i-1) + dx*(f(i+1)-f(i-1))
#                       = 0.5*( f(i) + f(i-1)*(1-dx) + f(i+1)*dx )
#
# dx*ex(1)/2 + (dx - 1.0)*ex(-1)/2 + ex(0)/2


Ex = Sum(Sum(Cdx[i+1]*Cpy[j+1]*f(i,j,0), (i, -1, 1)), (j, -1, 1)).doit()
#Ex = Sum(Sum(Cpx[i+1]*Cdy[j+1]*f(i,j,0), (i, -1, 1)), (j, -1, 1)).doit()

print(" ")
print(" ")
print("sum:")
print(Ex)

print(" ")
print(" ")
print("simplified sum:")
print(simplify(Ex))


