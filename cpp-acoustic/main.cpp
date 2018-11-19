#include "fem.h"

double CavitySolution(double x, double y, double z, double time){
  return cos(M_PI*x)*cos(M_PI*y)*cos(M_PI*z)*cos(sqrt(3)*M_PI*time);
};

int main(int argc, char **argv){

  Mesh *mesh;
  int k, n, sk=0;

  // read GMSH file
  mesh = ReadGmsh3d(argv[1]);
  int KblkV = 1, KblkS = 1, KblkU = 1;
  if(argc > 5){
    KblkV = atoi(argv[3]);
    KblkS = atoi(argv[4]);
    KblkU = atoi(argv[5]);
  }

  printf("Kblk (V,S,U) = %d,%d,%d\n",KblkV,KblkS,KblkU);
  
  // find element-element connectivity
  FacePair3d(mesh);
  
  // initialize arrays
  StartUp3d(mesh);
  printf("%d elements in mesh\n", mesh->K);

  
  // initialize OCCA info
  dfloat WaveInitOCCA3d(Mesh *mesh,int KblkV, int KblkS, int KblkU);

  dfloat dt = WaveInitOCCA3d(mesh,KblkV,KblkS,KblkU);
  printf("dt = %17.15f\n", dt);

  // default to cavity solution
  double (*uexptr)(double,double,double,double) = NULL;
  uexptr = &CavitySolution; // if domain = [-1,1]^3

  //  =========== field storage (dfloat) ===========

  dfloat *Q = (dfloat*) calloc(p_Nfields*mesh->K*p_Np, sizeof(dfloat));   // 4 fields
  dfloat *P = (dfloat*) calloc(p_Nfields*mesh->K*p_Np, sizeof(dfloat));

  // set u0
  WaveSetU0(mesh, Q, P, 0.0, uexptr); // interpolate
   
  double L2err, relL2err;
  
  // =============  run solver  ================

  // load data onto GPU
  WaveSetData3d(Q,P);

  printf("Order of the problem: p_N=%d, p_Np=%d, p_NMp=%d\n",p_N,p_Np,p_NMp);
    
  dfloat FinalTime = 0.1;
  if (argc > 2){
    FinalTime = atof(argv[2]);
  }
  
  printf("Running until time = %f\n",FinalTime);
  

  time_kernels(mesh); return 0;
  
  Wave_RK(mesh,FinalTime,dt); //run bb_WADG with M=1 and full-quadrature WADG  kernels
  
  //unload data from GPU
  WaveGetData3d(mesh, Q, P);
  compute_difference_Bern(mesh, Q, P, L2err, relL2err);
  
  printf("N = %d, Mesh size = %f, ndofs = %d, L2 difference at time %f = %6.6e\n",
	 p_N,mesh->hMax,mesh->K*p_Np,FinalTime,L2err);
  
  return 0;
  
}
