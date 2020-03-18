#include "boris_sdc.h"

#include <cmath> 
#include "../../tools/signum.h"

using toolbox::sign;

template<size_t D, size_t V>
void pic::BorisPusher<D,V>::push_container(
    pic::ParticleContainer& container, 
    double cfl, int push_type) 
{
  int nparts = container.size();

  // initialize pointers to particle arrays
  double* loc[3];
  for( int i=0; i<3; i++)
    loc[i] = &( container.loc(i,0) );

  double* vel[3];
  for( int i=0; i<3; i++)
    vel[i] = &( container.vel(i,0) );


  double ex0 = 0.0, ey0 = 0.0, ez0 = 0.0;
  double bx0 = 0.0, by0 = 0.0, bz0 = 0.0;

  // make sure E and B tmp arrays are of correct size
  if(container.Epart.size() != (size_t)3*nparts)
    container.Epart.resize(3*nparts);
  if(container.Bpart.size() != (size_t)3*nparts)
    container.Bpart.resize(3*nparts);

  double *ex, *ey, *ez, *bx, *by, *bz;
  ex = &( container.Epart[0*nparts] );
  ey = &( container.Epart[1*nparts] );
  ez = &( container.Epart[2*nparts] );

  bx = &( container.Bpart[0*nparts] );
  by = &( container.Bpart[1*nparts] );
  bz = &( container.Bpart[2*nparts] );

  // loop over particles
  int n1 = 0;
  int n2 = nparts;

  double u0, v0, w0;
  double u1, v1, w1;
  double g, f;

  double c = cfl;
  double cinv = 1.0/c;

  // charge (sign only)
  double qm = sign(container.q);

  // add division by m_s to simulate multiple species

  int K = container.K
  int M = container.M
  int k = container.k
  int m = container.m

  double ck [3];

  //TODO: SIMD
  //--------------------------------------------------
  // Boris-SDC algorithm

  // Position update
  if(push_type == 0)  {
    for(int n=n1; n<n2; n++) {


      double sumSX [3] = {0,0,0}; 
      for(int l=1; l<m+1; l++) {
        f = lorentz(loc[k][l][:][n],loc[k][l][:][n]);
        sumSX[0] += SX[m+1][l]*f[0];
        sumSX[1] += SX[m+1][l]*f[1];
        sumSX[2] += SX[m+1][l]*f[2];
      }


      double sumSQ [3] = {0,0,0}; 
      for(int l=1; l<M+1; l++) {
        f = lorentz(loc[k][l][:][n],loc[k][l][:][n]);
        sumSQ[0] += SQ[m+1][l]*f[0];
        sumSQ[1] += SQ[m+1][l]*f[1];
        sumSQ[2] += SQ[m+1][l]*f[2];
      }


      for(size_t i=0; i<D; i++)
        loc[k+1][m+1][i][n] = loc[k+1][m][i][n] + dtau[m+1]*vel[k+1][m+1][i][n]
        loc[k+1][m+1][i][n] += sumSX[i]
        loc[k+1][m+1][i][n] += sumSQ[i]
    }

  // Velocity update
  if(push_type == 1)  {
    for(int n=n1; n<n2; n++) {

      double sumS [3] = {0,0,0}; 
      for(int l=1; l<M+1; l++) {
        f = lorentz(loc[k][l][:][n],loc[k][l][:][n]);
        sumS[0] += S[m+1][l]*f[0];
        sumS[1] += S[m+1][l]*f[1];
        sumS[2] += S[m+1][l]*f[2];
      }

      f1 = lorentz(loc[k][m+1][:][n],loc[k][l][:][n]);
      f2 = lorentz(loc[k][m][:][n],loc[k][l][:][n]);
      ck = -1/2*(f1+f2) + dtau[m] * sumS

      // read particle-specific fields
      ex0 = 0.5*dtau[m] * (ex[n]*qm + ck[0]);
      ey0 = 0.5*dtau[m] * (ey[n]*qm + ck[1]);
      ez0 = 0.5*dtau[m] * (ez[n]*qm + ck[2]);

      bx0 = bx[n]*(0.5*qm*cinv);
      by0 = by[n]*(0.5*qm*cinv);
      bz0 = bz[n]*(0.5*qm*cinv);


      // first half electric acceleration
      u0 = c*vel[k][m][0][n] + ex0;
      v0 = c*vel[k][m][1][n] + ey0;
      w0 = c*vel[k][m][2][n] + ez0;

      // first half magnetic rotation
      g = c/sqrt(c*c + u0*u0 + v0*v0 + w0*w0);
      bx0 *= g;
      by0 *= g;
      bz0 *= g;

      f = 2.0/(1.0 + bx0*bx0 + by0*by0 + bz0*bz0);
      u1 = (u0 + v0*bz0 - w0*by0)*f;
      v1 = (v0 + w0*bx0 - u0*bz0)*f;
      w1 = (w0 + u0*by0 - v0*bx0)*f;

      // second half of magnetic rotation & electric acceleration
      u0 = u0 + v1*bz0 - w1*by0 + ex0;
      v0 = v0 + w1*bx0 - u1*bz0 + ey0;
      w0 = w0 + u1*by0 - v1*bx0 + ez0;

      // normalized 4-velocity advance
      vel[k][m][0][n] = u0*cinv;
      vel[k][m][1][n] = v0*cinv;
      vel[k][m][2][n] = w0*cinv;
    }

  }
}



//--------------------------------------------------
// explicit template instantiation

template class pic::BorisPusher<1,3>; // 1D3V
template class pic::BorisPusher<2,3>; // 2D3V
template class pic::BorisPusher<3,3>; // 3D3V

