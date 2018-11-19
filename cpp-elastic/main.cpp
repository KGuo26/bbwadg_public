#include "fem.h"

double PulseInitialCondition(double x, double y, double z, double time){
  double rad = (x*x + y*y + z*z);
  double a = 1.0;
  return exp(-a*a*rad);
};


double WaveWeight(double x, double y, double z){
  double k = 3.0;
  double a = 0.5;
  return a*cos(k*M_PI*x)*cos(k*M_PI*y)*cos(k*M_PI*z);
};


int main(int argc, char **argv){
  
  printf("running heterogeneous subelem main\n");

  Mesh *mesh;
  int k,n, sk=0;

  // read GMSH file
  mesh = ReadGmsh3d(argv[1]);

  int KblkV = 1, KblkS = 1, KblkU = 1, KblkQ = 1, KblkQf = 1;
  if(argc >= 8){
    KblkV = atoi(argv[3]);
    KblkS = atoi(argv[4]);
    KblkU = atoi(argv[5]);
  }
  printf("for N = %d, Kblk (V,S,U) = %d,%d,%d\n",
  	 p_N,KblkV,KblkS,KblkU);

  FacePair3d(mesh);
  StartUp3d(mesh);

  printf("%d elements in mesh\n", mesh->K);

  // initialize OCCA info
  dfloat WaveInitOCCA3d(Mesh *mesh,int KblkV, int KblkS, int KblkU, int KblkQ, int KblkQf);
  dfloat dt = WaveInitOCCA3d(mesh,KblkV,KblkS,KblkU,KblkQ,KblkQf);
  printf("dt = %17.15f\n", dt);

  InitQuadratureArrays(mesh); // initialize quadrature-based geofacs on host

  double (*wptr)(double,double,double) = &WaveWeight;
  
  AdaptiveM(mesh,wptr);

  BB_mult(mesh);

  BB_projection(mesh);

  InitWADG_subelem(mesh,wptr);

  printf("initialized wadg subelem\n");

  //  =========== field storage (dfloat) ===========

  printf("Number of field = %d\n",p_Nfields);
  dfloat *Q = (dfloat*) calloc(p_Nfields*mesh->K*p_Np, sizeof(dfloat));   // 9 fields
  dfloat *P = (dfloat*) calloc(p_Nfields*mesh->K*p_Np, sizeof(dfloat));


  
  double (*uexptr)(double,double,double,double)= NULL;
  uexptr = &PulseInitialCondition;
  int field = 0;
  WaveSetU0P0(mesh,Q,P,0.0,field,uexptr); 
  
  
  // =============  run solver  ================
  
  // load data onto GPU
  printf("Loading data onto GPU\n");
  WaveSetData3d(Q,P);
  
  time_kernels_elas(mesh); return 0;
  
  dfloat FinalTime = 1.5;
  if (argc > 2){
    FinalTime = atof(argv[2]);
  }

  printf("Running until time = %f\n",FinalTime);

  //  FinalTime = 1* dt;
  Wave_RK(mesh,FinalTime,dt,1); // run w/planar elems and WADG update

  printf("Unloading data from GPU\n");
  WaveGetData3d(mesh, Q, P);  // unload data from GPU
  
  
  printf("Writing data to GMSH\n");
  writeVisToGMSH("meshes/p.msh",mesh,Q,0,p_Nfields);

  double L2err, relL2err;
  compute_error_adaptive(mesh,Q,P,L2err,relL2err);

  printf("N = %d, Mesh size = %f, ndofs = %d, L2 error at time %f = %6.6e\n", p_N,mesh->hMax,mesh->K*p_Np,FinalTime,L2err);   

  /*    
  for(int i = 0; i < p_Np; ++i){
    printf("Q[%d]=%f\n", i , Q[i]);
    printf("P[%d]=%f\n", i,  P[i]);
  }
  */
  
  return 0;

  exit(0);
}
