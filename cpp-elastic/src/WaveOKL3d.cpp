#include <stdio.h>
#include "fem.h"

#include <occa.hpp>

// switches b/w nodal and bernstein bases
#define USE_SKEW 1
#define USE_SLICE_LIFT 0 // switch on for faster behavior if N > 6

int ngeo, nvgeo, nfgeo; // number of geometric factors

// OCCA device
occa::device device;
occa::kernelInfo dgInfo;

// OCCA array for geometric factors
occa::memory c_vgeo;
occa::memory c_fgeo;

// OCCA array for index of positive trace variables wrt face nodes
occa::memory c_vmapP;
occa::memory c_Fmask;

// rhs for RK kernels
occa::memory c_rhsQ;
occa::memory c_resQ;
occa::memory c_Q;

occa::memory c_rhsP;
occa::memory c_resP;
occa::memory c_P;


// OCCA arrays for nodal derivative/lift matrices
occa::memory c_Dr;
occa::memory c_Ds;
occa::memory c_Dt;
occa::memory c_LIFT;

// Bernstein derivative operators
occa::memory c_Dvals4;
occa::memory c_D_ids1;
occa::memory c_D_ids2;
occa::memory c_D_ids3;
occa::memory c_D_ids4;

// Bernstein LIFT decomposition
occa::memory c_EEL_vals;
occa::memory c_EEL_ids;
occa::memory c_L0_vals;
occa::memory c_L0_ids;
occa::memory c_cEL;
occa::memory c_slice_ids; // permutation of rows

// nodal kernels
occa::kernel rk_volume;
occa::kernel rk_surface;

// bernstein kernels
occa::kernel rk_volume_bern;
occa::kernel rk_surface_bern;

// used for both nodal and Bernstein
occa::kernel rk_update;

// add kernels for BB basis elasity
occa::kernel rk_volume_bern_elas;
occa::kernel rk_surface_bern_elas;
occa::kernel rk_update_bern_WADG_elas;
occa::kernel rk_update_bern_full_WADG_elas;
occa::kernel rk_update_bern_elas;

// WADG for heterogeneous media w/planar elems
occa::memory c_Vq_reduced;
occa::memory c_Pq_reduced;

occa::memory c_Vq_BB;
occa::memory c_Pq_BB;

// transversely isotropic media
occa::memory c_rhoq;
occa::memory c_lambdaq;
occa::memory c_muq;
occa::memory c_c11;
occa::memory c_c12;

occa::memory c_rho_BB;
occa::memory c_lambda_BB;
occa::memory c_mu_BB;

// BB projection and multiplication
occa::memory c_col_val;
occa::memory c_col_id;
occa::memory c_L_id;
occa::memory c_ENMT_val;
occa::memory c_ENMT_id;
occa::memory c_ENM_val;
occa::memory c_ENM_id;
occa::memory c_co;
occa::memory c_E;
occa::memory c_ENMT_index;

// ricker ptsrc
occa::memory c_fsrc;
occa::memory c_fsrc_BB;

occa::kernel rk_volume_elas;
occa::kernel rk_surface_elas;
occa::kernel rk_update_elas;

// block sizes for optimization of diff kernels
int KblkV, KblkS, KblkU, KblkQ, KblkQf;

// runtime counters
double timeV = 0.0, timeS = 0.0, timeU=0.0, timeQ = 0.0, timeQf = 0.0;

template <typename T>
void diagnose_array(const char *message, occa::memory &c_a, int N){

  device.finish();

  T *h_a = (T*) calloc(N, sizeof(T));

  c_a.copyTo(h_a, N*sizeof(T));

  dfloat suma = 0;
  for(int n=0;n<N;++n){
    suma += h_a[n];
  }

  printf("%s: sum = %17.15g\n", message, suma);

  free(h_a);
}

void InitWADG_subelem(Mesh *mesh,double(*c2_ptr)(double,double,double)){

  int Nq = mesh->Nq;

  // ======= use reduced strength quadrature for WADG update

  int Nq1D = p_N+1; // N+1 for GQ(1,0) in vert, N+1 for GQ(0,0)
  int Nq2 = Nq1D*Nq1D;
  int Nq3 = Nq2*Nq1D;

  VectorXd rq,sq,tq,wq;
  //  if (0){
  if (2*p_N+1 <= 15){
    tet_cubature(min(21,2*p_N+1),rq,sq,tq,wq); // 21 = max quadrature degree
  }else{

    // default to TP quadrature for now
    VectorXd a1D,wa,c1D,wc;
    JacobiGQ(p_N,0,0,a1D,wa);
    JacobiGQ(p_N,2,0,c1D,wc);
    rq.resize(Nq3);    sq.resize(Nq3);
    tq.resize(Nq3);    wq.resize(Nq3);
    int sk = 0;
    for (int k = 0; k < Nq1D; ++k){
      for (int i = 0; i < Nq1D; ++i){
        for (int j = 0; j < Nq1D; ++j){
          rq(sk) = a1D(i); sq(sk) = a1D(j);
	  tq(sk) = c1D(k); wq(sk) = wa(i)*wa(j)*wc(k);
          ++sk;
        }
      }
    }
  }
  MatrixXd Vqtmp = Vandermonde3D(p_N,rq,sq,tq);
  MatrixXd Vq_reduced = mrdivide(Vqtmp,mesh->V);
 
  
  // Vq_BB
  MatrixXd Vq_BB = Vq_reduced * mesh->VB;
 
  // array of wavefield at quadrature points
  MatrixXd rhoq(Vq_reduced.rows(),mesh->K);
  MatrixXd lambdaq(Vq_reduced.rows(),mesh->K);
  MatrixXd muq(Vq_reduced.rows(),mesh->K);
  MatrixXd c11(Vq_reduced.rows(),mesh->K);
  MatrixXd c12(Vq_reduced.rows(),mesh->K);

  MatrixXd fsrcq(Vq_reduced.rows(),mesh->K);

  for (int e = 0; e < mesh->K; ++e){

    // locally interpolate to cubature points
    VectorXd xq(Vq_reduced.rows()), yq(Vq_reduced.rows()), zq(Vq_reduced.rows());
    xq = Vq_reduced*(mesh->x.col(e));
    yq = Vq_reduced*(mesh->y.col(e));
    zq = Vq_reduced*(mesh->z.col(e));

    for (int i = 0; i < Vq_reduced.rows(); ++i){

      double weight = (*c2_ptr)(xq(i),yq(i),zq(i));

      double mu = 1; double lambda = 1;
      rhoq(i,e) = 1.0;
      lambdaq(i,e) = lambda;
      muq(i,e) = mu + weight; // constant mu
      /*
#if 0
      double A = 2*mu+lambda;
      double B = lambda;

      // transverse isotropy
      c11(i,e) = A + weight;
      c12(i,e) = B + weight;
      if (zq(i) > 0){ // vertical part
	c11(i,e) *= 1.0/3.0; // 2*mu + lambda
	c12(i,e) *= 0.5;
      }
#else
      */
      // isotropic but discontinuous
      double BB = 2.0;
      if (zq(i) < 0){
        muq(i,e) = BB + weight;
        lambdaq(i,e) = BB;
      }

      //#endif
      // smoothed ricker pulse
      double a = 100.0;
      double x0 = 0.0;
      double y0 = 0.0;
      double z0 = .1;
      double dx = xq(i) - x0;
      double dy = yq(i) - y0;
      double dz = zq(i) - z0;
      double r2 = dx*dx + dy*dy + dz*dz;
      fsrcq(i,e) = exp(-a*a*r2);
    }
  }

  
#if 1 // take local average of coeffs

  for (int e = 0; e < mesh->K; ++e){
    double muavg = wq.dot(muq.col(e)) / wq.sum();
    muq.col(e).fill(muavg);
  }
#endif

  /*
  cout<<"lambda="<<endl<<lambdaq<<endl;
  cout<<"rho="<<endl<<rhoq<<endl;
  */  
  //cout<<"mu="<<endl<<muq<<endl;
  

  dgInfo.addDefine("p_tau_v",1.0); // velocity penalty
  dgInfo.addDefine("p_tau_s",1.0);

  MatrixXd invM = mesh->V*mesh->V.transpose();
  MatrixXd Pq_reduced = invM*Vq_reduced.transpose()*wq.asDiagonal();
  printf("size of Pq: %d * %d\n", Pq_reduced.rows(),Pq_reduced.cols());
  MatrixXd Pq_BB = mesh->invVB * Pq_reduced;

  
  dgInfo.addDefine("p_Nq_reduced",Vq_reduced.rows()); // for update step quadrature

  // not used in WADG subelem - just to compile other kernels
  dgInfo.addDefine("p_NfqNfaces",mesh->Nfq * p_Nfaces); // surf quadrature
  dgInfo.addDefine("p_Nq",mesh->Nq); // vol quadrature
  dgInfo.addDefine("p_Nfq",mesh->Nfq); // surf quadrature for one face

  setOccaArray(rhoq,c_rhoq);
  setOccaArray(lambdaq,c_lambdaq);
  setOccaArray(muq,c_muq);
  setOccaArray(c11,c_c11);
  setOccaArray(c12,c_c12);

  // coefficients for BB basis
  setOccaArray(mesh->rho_BB, c_rho_BB);
  setOccaArray(mesh->lambda_BB, c_lambda_BB);
  setOccaArray(mesh->mu_BB, c_mu_BB);
  
  // smoothed ricker src
  fsrcq *= 1.0/fsrcq.array().abs().maxCoeff(); // normalize to 1
  MatrixXd fsrc = Pq_reduced * fsrcq;
  MatrixXd fsrc_BB = mesh->invVB * fsrc; 
  
  setOccaArray(fsrc_BB,c_fsrc_BB);
  setOccaArray(fsrc,c_fsrc);
  setOccaArray(Pq_reduced,c_Pq_reduced);
  setOccaArray(Vq_reduced,c_Vq_reduced);
  setOccaArray(Pq_BB,c_Pq_BB);
  setOccaArray(Vq_BB,c_Vq_BB);
  setOccaArray(mesh->col_val, c_col_val);
  setOccaIntArray(mesh->col_id, c_col_id);
  setOccaIntArray(mesh->L_id, c_L_id);
  setOccaArray(mesh->ENMT_val, c_ENMT_val);
  setOccaIntArray(mesh->ENMT_id, c_ENMT_id);
  setOccaArray(mesh->ENM_val, c_ENM_val);
  setOccaIntArray(mesh->ENM_id, c_ENM_id);
  setOccaArray(mesh->co, c_co);
  setOccaArray(mesh->E, c_E);
  setOccaIntArray(mesh->ENMT_index, c_ENMT_index);
  
  // ======== build kernels

  std::string src = "okl/ElasKernelsWADG.okl";
  printf("Building heterogeneous wave propagation WADG kernel from %s\n",src.c_str());
  rk_update_elas  = device.buildKernelFromSource(src.c_str(), "rk_update_const_elas", dgInfo);
  rk_volume_elas  = device.buildKernelFromSource(src.c_str(), "rk_volume_elas", dgInfo);
  rk_surface_elas = device.buildKernelFromSource(src.c_str(), "rk_surface_elas", dgInfo);

  rk_volume_bern_elas  = device.buildKernelFromSource(src.c_str(),"rk_volume_bern_elas", dgInfo);
  rk_surface_bern_elas = device.buildKernelFromSource(src.c_str(),"rk_surface_bern_elas", dgInfo);
  rk_update_bern_elas  = device.buildKernelFromSource(src.c_str(),"rk_update_bern_const_elas", dgInfo);
}


// set occa array:: cast to dfloat.
void setOccaArray(MatrixXd A, occa::memory &c_A){
  int r = A.rows();
  int c = A.cols();
  dfloat *f_A = (dfloat*)malloc(r*c*sizeof(dfloat));
  Map<MatrixXdf >(f_A,r,c) = A.cast<dfloat>();
  c_A = device.malloc(r*c*sizeof(dfloat),f_A);
  free(f_A);
}

// set occa array:: cast to dfloat
void setOccaIntArray(MatrixXi A, occa::memory &c_A){
  int r = A.rows();
  int c = A.cols();
  int *f_A = (int*)malloc(r*c*sizeof(int));
  Map<MatrixXi >(f_A,r,c) = A;
  c_A = device.malloc(r*c*sizeof(int),f_A);
  free(f_A);
}


dfloat WaveInitOCCA3d(Mesh *mesh, int KblkVin, int KblkSin,
		      int KblkUin, int KblkQin, int KblkQfin){

  KblkV = KblkVin;
  KblkS = KblkSin;
  KblkU = KblkUin;
  KblkQ = KblkQin;
  KblkQf = KblkQfin;

  occa::printAvailableDevices();

  //device.setup("mode = OpenCL, platformID = 0, deviceID = 0");
  //device.setup("mode = OpenMP, platformID = 0, deviceID = 0");
  //device.setup("mode = Serial");
  device.setup("mode = CUDA, platformID = 0, deviceID = 0");

  //device.setCompiler("nvcc"); device.setCompilerFlags("--use_fast_math"); device.setCompilerFlags("--fmad=true");

  printf("KblkV = %d, KblkS = %d, KblkU = %d, KblkQ (skew only) = %d\n",
	 KblkV,KblkS,KblkU,KblkQ);

  int K = mesh->K;
  int NpK = K*p_Np;

  int sz;

  // nodal operators
  sz = p_Np*p_Np*sizeof(dfloat);
  dfloat *f_Dr = (dfloat*)malloc(p_Np*p_Np*sizeof(dfloat));
  dfloat *f_Ds = (dfloat*)malloc(p_Np*p_Np*sizeof(dfloat));
  dfloat *f_Dt = (dfloat*)malloc(p_Np*p_Np*sizeof(dfloat));
  dfloat *f_LIFT = (dfloat*)malloc(p_Np*p_Nfp*p_Nfaces*sizeof(dfloat));

  // alternative way to cast matrices to dfloat
  Map<MatrixXdf >(f_Dr,p_Np,p_Np) = mesh->Dr.cast<dfloat>();
  Map<MatrixXdf >(f_Ds,p_Np,p_Np) = mesh->Ds.cast<dfloat>();
  Map<MatrixXdf >(f_Dt,p_Np,p_Np) = mesh->Dt.cast<dfloat>();
  Map<MatrixXdf >(f_LIFT,p_Np,p_Nfp*p_Nfaces) = mesh->LIFT.cast<dfloat>();

  c_Dr = device.malloc(sz,f_Dr);
  c_Ds = device.malloc(sz,f_Ds);
  c_Dt = device.malloc(sz,f_Dt);
  c_LIFT = device.malloc(p_Np*p_Nfp*p_Nfaces*sizeof(dfloat),f_LIFT);

  c_Fmask = device.malloc(p_Nfp*p_Nfaces*sizeof(int),mesh->FmaskC[0]);

  // ==================== BERNSTEIN STUFF ==========================

  // bernstein Dmatrices (4 entries per row)
  sz = 4*p_Np*sizeof(int);
  c_Dvals4 = device.malloc(4*p_Np*sizeof(dfloat), mesh->D_vals[0]);

  // barycentric deriv indices organized for ILP
  int *D_ids1 = (int*) malloc(p_Np*4*sizeof(int));
  int *D_ids2 = (int*) malloc(p_Np*4*sizeof(int));
  int *D_ids3 = (int*) malloc(p_Np*4*sizeof(int));
  int *D_ids4 = (int*) malloc(p_Np*4*sizeof(int));
  for(int i = 0; i < p_Np; ++i){
    D_ids1[4*i + 0] = mesh->D1_ids[i][0];
    D_ids1[4*i + 1] = mesh->D2_ids[i][0];
    D_ids1[4*i + 2] = mesh->D3_ids[i][0];
    D_ids1[4*i + 3] = mesh->D4_ids[i][0];

    D_ids2[4*i + 0] = mesh->D1_ids[i][1];
    D_ids2[4*i + 1] = mesh->D2_ids[i][1];
    D_ids2[4*i + 2] = mesh->D3_ids[i][1];
    D_ids2[4*i + 3] = mesh->D4_ids[i][1];

    D_ids3[4*i + 0] = mesh->D1_ids[i][2];
    D_ids3[4*i + 1] = mesh->D2_ids[i][2];
    D_ids3[4*i + 2] = mesh->D3_ids[i][2];
    D_ids3[4*i + 3] = mesh->D4_ids[i][2];

    D_ids4[4*i + 0] = mesh->D1_ids[i][3];
    D_ids4[4*i + 1] = mesh->D2_ids[i][3];
    D_ids4[4*i + 2] = mesh->D3_ids[i][3];
    D_ids4[4*i + 3] = mesh->D4_ids[i][3];
  }
  c_D_ids1 = device.malloc(sz,D_ids1);
  c_D_ids2 = device.malloc(sz,D_ids2);
  c_D_ids3 = device.malloc(sz,D_ids3);
  c_D_ids4 = device.malloc(sz,D_ids4);


  dfloat *h_EEL_vals = (dfloat*) malloc(p_Np*mesh->EEL_nnz*sizeof(dfloat));
  int *h_EEL_ids = (int*) malloc(p_Np*mesh->EEL_nnz*sizeof(int));
  for (int i = 0; i < p_Np; ++i){
    for (int j = 0; j < mesh->EEL_nnz; ++j){
      h_EEL_vals[i + j*p_Np] = mesh->EEL_vals[i][j];
      h_EEL_ids[i + j*p_Np] = mesh->EEL_ids[i][j];
    }
  }

  int L0_nnz = min(p_Nfp,7);
  mesh->L0_nnz = L0_nnz;
  dfloat *h_L0_vals = (dfloat*) malloc(p_Nfp*L0_nnz*sizeof(dfloat));
  int *h_L0_ids = (int*) malloc(p_Nfp*L0_nnz*sizeof(int));
  for (int i = 0; i < p_Nfp; ++i){
    for (int j = 0; j < L0_nnz; ++j){
      h_L0_vals[i + j*p_Nfp] = mesh->L0_vals[i][j];
      h_L0_ids[i + j*p_Nfp] = mesh->L0_ids[i][j];
    }
  }

  c_L0_vals = device.malloc(p_Nfp*L0_nnz*sizeof(dfloat),h_L0_vals);
  c_L0_ids = device.malloc(p_Nfp*L0_nnz*sizeof(int),h_L0_ids);

#if (USE_SLICE_LIFT)  // should use for N > 5 (faster)
  // store reduction matrices for orders 0,...,N
  setOccaIntArray(mesh->EEL_id_vec,c_EEL_ids);
  setOccaArray(mesh->EEL_val_vec,c_EEL_vals);
#else
  c_EEL_vals = device.malloc(p_Np*mesh->EEL_nnz*sizeof(dfloat),h_EEL_vals);
  c_EEL_ids = device.malloc(p_Np*mesh->EEL_nnz*sizeof(int),h_EEL_ids);
#endif

  //cout << "cEL = " << endl << mesh->cEL << endl;
  setOccaArray(mesh->cEL,c_cEL); // for slice-by-slice kernel

  int *h_slice_ids = (int*) malloc(p_Np*4*sizeof(int));
  for (int i = 0; i < p_Np; ++i){
    h_slice_ids[4*i+0] = mesh->slice_ids(i,0);
    h_slice_ids[4*i+1] = mesh->slice_ids(i,1);
    h_slice_ids[4*i+2] = mesh->slice_ids(i,2);
    h_slice_ids[4*i+3] = mesh->slice_ids(i,3);
  }
  //  setOccaIntArray(vol_ids,c_vol_ids); // arranged for int4 storage
  c_slice_ids = device.malloc(p_Np*4*sizeof(int),h_slice_ids);



  // =====================  geofacs ==================================

  double drdx, dsdx, dtdx;
  double drdy, dsdy, dtdy;
  double drdz, dsdz, dtdz, J;

  int sk = 0;
  int skP = -1;
  double *nxk = (double*) calloc(mesh->Nfaces,sizeof(double));//BuildVector(mesh->Nfaces);
  double *nyk = (double*) calloc(mesh->Nfaces,sizeof(double));//BuildVector(mesh->Nfaces);
  double *nzk = (double*) calloc(mesh->Nfaces,sizeof(double));//BuildVector(mesh->Nfaces);
  double *sJk = (double*) calloc(mesh->Nfaces,sizeof(double));//BuildVector(mesh->Nfaces);

  // [JC] packed geo + surface
  nvgeo = 9; // rst/xyz
  nfgeo = 4; // Fscale, (3)nxyz,
  ngeo = nfgeo*p_Nfaces + nvgeo; // nxyz + tau + Fscale (faces), G'*G (volume)
  dfloat *geo = (dfloat*) malloc(mesh->K*ngeo*sizeof(dfloat));
  dfloat *vgeo = (dfloat*) malloc(K*nvgeo*sizeof(dfloat));
  dfloat *fgeo = (dfloat*) malloc(K*nfgeo*p_Nfaces*sizeof(dfloat));

  dfloat FscaleMax = 0.f;
  for(int k=0;k<mesh->K;++k){

    GeometricFactors3d(mesh, k,
		       &drdx, &dsdx, &dtdx,
		       &drdy, &dsdy, &dtdy,
		       &drdz, &dsdz, &dtdz, &J);

    Normals3d(mesh, k, nxk, nyk, nzk, sJk);

    vgeo[k*nvgeo + 0] = drdx;    vgeo[k*nvgeo + 1] = drdy;    vgeo[k*nvgeo + 2] = drdz;
    vgeo[k*nvgeo + 3] = dsdx;    vgeo[k*nvgeo + 4] = dsdy;    vgeo[k*nvgeo + 5] = dsdz;
    vgeo[k*nvgeo + 6] = dtdx;    vgeo[k*nvgeo + 7] = dtdy;    vgeo[k*nvgeo + 8] = dtdz;

#if 0
    if (k==0){
      printf("rxyz = %f, %f, %f\n",drdx,drdy,drdz);
      printf("sxyz = %f, %f, %f\n",dsdx,dsdy,dsdz);
      printf("txyz = %f, %f, %f\n",dtdx,dtdy,dtdz);
      printf("J = %f\n\n",J);
    }
#endif

    for(int f=0;f<mesh->Nfaces;++f){

      dfloat Fscale = sJk[f]/J; //sJk[f]/(2.*J);
      dfloat nx = nxk[f];
      dfloat ny = nyk[f];
      dfloat nz = nzk[f];

      // for dt
      FscaleMax = max(FscaleMax,Fscale);

      fgeo[k*nfgeo*p_Nfaces + f*nfgeo + 0] = Fscale; // Fscale
      fgeo[k*nfgeo*p_Nfaces + f*nfgeo + 1] = nx;
      fgeo[k*nfgeo*p_Nfaces + f*nfgeo + 2] = ny;
      fgeo[k*nfgeo*p_Nfaces + f*nfgeo + 3] = nz;
    }
  }
  int *h_vmapP = (int*) malloc(mesh->K*p_Nfp*p_Nfaces*sizeof(int));
  for (int e = 0; e < mesh->K; ++e){
    for (int i = 0; i < p_Nfp*p_Nfaces; ++i){
      int f = i/p_Nfp;
      int idP = mesh->vmapP[i + p_Nfp*p_Nfaces*e];

      // correct vmapP for Nfields > 1
      int eNbr = mesh->EToE[e][f];
      idP -= p_Np*eNbr; // decrement
      idP += p_Np*p_Nfields*eNbr; // re-increment

      h_vmapP[i+p_Nfp*p_Nfaces*e] = idP;
    }
  }

  // storage for solution variables
  dfloat *f_Q    = (dfloat*) calloc(mesh->K*p_Nfields*p_Np, sizeof(dfloat));
  dfloat *f_resQ = (dfloat*) calloc(mesh->K*p_Nfields*p_Np, sizeof(dfloat));
  dfloat *f_rhsQ = (dfloat*) calloc(mesh->K*p_Nfields*p_Np, sizeof(dfloat));

  c_Q    = device.malloc(sizeof(dfloat)*mesh->K*p_Np*p_Nfields, f_Q);
  c_resQ = device.malloc(sizeof(dfloat)*mesh->K*p_Np*p_Nfields, f_resQ);
  c_rhsQ = device.malloc(sizeof(dfloat)*mesh->K*p_Np*p_Nfields, f_rhsQ);

  dfloat *f_P    = (dfloat*) calloc(mesh->K*p_Nfields*p_Np, sizeof(dfloat));
  dfloat *f_resP = (dfloat*) calloc(mesh->K*p_Nfields*p_Np, sizeof(dfloat));
  dfloat *f_rhsP = (dfloat*) calloc(mesh->K*p_Nfields*p_Np, sizeof(dfloat));

  c_P    = device.malloc(sizeof(dfloat)*mesh->K*p_Np*p_Nfields, f_P);
  c_resP = device.malloc(sizeof(dfloat)*mesh->K*p_Np*p_Nfields, f_resP);
  c_rhsP = device.malloc(sizeof(dfloat)*mesh->K*p_Np*p_Nfields, f_rhsP);



  c_vgeo = device.malloc(mesh->K*nvgeo*sizeof(dfloat), vgeo);
  c_fgeo = device.malloc(mesh->K*nfgeo*p_Nfaces*sizeof(dfloat), fgeo);
  c_vmapP  = device.malloc(p_Nfp*p_Nfaces*mesh->K*sizeof(int),h_vmapP);

  // build kernels
  if (sizeof(dfloat)==8){
    dgInfo.addDefine("USE_DOUBLE", 1);
  }else{
    dgInfo.addDefine("USE_DOUBLE", 0);
  }

  dgInfo.addDefine("p_EEL_size",mesh->EEL_val_vec.rows());
  dgInfo.addDefine("p_EEL_nnz",mesh->EEL_nnz);
  dgInfo.addDefine("p_L0_nnz",min(p_Nfp,7)); // max 7 nnz with L0 matrix

  printf("p_Nfields = %d\n",p_Nfields);
  dgInfo.addDefine("p_Nfields",      p_Nfields); // wave equation

  dgInfo.addDefine("p_N",      p_N);
  dgInfo.addDefine("p_KblkV",  KblkV);
  dgInfo.addDefine("p_KblkS",  KblkS);
  dgInfo.addDefine("p_KblkU",  KblkU);
  dgInfo.addDefine("p_KblkQ",  KblkQ);
  dgInfo.addDefine("p_KblkQf",  KblkQf);

  dgInfo.addDefine("p_Np",      p_Np);
  dgInfo.addDefine("p_Nfp",     p_Nfp);
  dgInfo.addDefine("p_Nfaces",  p_Nfaces);
  dgInfo.addDefine("p_NfpNfaces",     p_Nfp*p_Nfaces);

  dgInfo.addDefine("p_NMp", p_NMp);
  dgInfo.addDefine("p_1p",4);
  
  int p_max = max(p_Np , 6);
  dgInfo.addDefine("p_max", p_max);
  // [JC] max threads
  dgInfo.addDefine("p_ceilNq",min(512,mesh->Nq));
  int T = max(p_Np,p_Nfp*p_Nfaces);
  dgInfo.addDefine("p_T",T);
  int Tq = max(p_Np,mesh->Nfq*p_Nfaces);
  dgInfo.addDefine("p_Tq",Tq);

  dgInfo.addDefine("p_Nvgeo",nvgeo);
  dgInfo.addDefine("p_Nfgeo",nfgeo);


  std::string src = "okl/WaveKernels.okl";
  std::cout << "using src = " << src.c_str() << std::endl;

#if USE_BERN
  printf("Building Bernstein kernels from %s\n",src.c_str());
  // bernstein kernels
  rk_volume_bern  = device.buildKernelFromSource(src.c_str(), "rk_volume_bern", dgInfo);

  printf("building rk_surface_bern from %s\n",src.c_str());

#if USE_SLICE_LIFT
  rk_surface_bern = device.buildKernelFromSource(src.c_str(), "rk_surface_bern_slice", dgInfo);
  printf("using slice-by-slice bern surface kernel; more efficient for N > 6\n");
#else
  printf("using non-optimal bern surface kernel; more efficient for N < 6\n");
  rk_surface_bern = device.buildKernelFromSource(src.c_str(), "rk_surface_bern", dgInfo);
#endif
#endif
  // nodal kernels
  rk_volume  = device.buildKernelFromSource(src.c_str(), "rk_volume", dgInfo);
  rk_surface = device.buildKernelFromSource(src.c_str(), "rk_surface", dgInfo);
  rk_update  = device.buildKernelFromSource(src.c_str(), "rk_update", dgInfo);

  // estimate dt. may wish to replace with trace inequality constant
  dfloat CN = (p_N+1)*(p_N+3)/3.0;
  dfloat dt = .25/(CN*FscaleMax);

  return (dfloat) dt;
}

// compute quadrature nodes + error
void compute_error(Mesh *mesh, double time, dfloat *Q,
		   double(*uexptr)(double,double,double,double),
		   double &L2err, double &relL2err){

  // compute error
  L2err = 0.0;
  double L2norm = 0.0;
  int kChunk = max(mesh->K/10,1);
  for(int k=0;k<mesh->K;++k){

    //double J = mesh->J(0,k); // assuming J = constant (planar tet)
    for(int i = 0; i < mesh->Nq; ++i){

      double J;
      if (mesh->Jq.rows()>0){ // if wadg quadrature initialized
	J= mesh->Jq(i,k);
      }else{
	J = mesh->J(0,k);
      }

      // interp to cubature nodes
      double x = 0.0; double y = 0.0; double z = 0.0; double uq = 0.0;
      for(int j=0;j<p_Np;++j){

	double Vq = mesh->Vq(i,j);
        x += Vq*mesh->x(j,k);
        y += Vq*mesh->y(j,k);
        z += Vq*mesh->z(j,k);
        uq += Vq*Q[j+p_Np*p_Nfields*k]; // get field value for pressure
      }

      double uex = (*uexptr)(x,y,z,(double)time);
      double err = uq-uex;

      L2err += err*err*mesh->wq(i)*J;
      L2norm += uex*uex*mesh->wq(i)*J;
    }
  }
  L2err = sqrt(L2err);
  relL2err = sqrt(L2err)/sqrt(L2norm);

  return;

}

// times main_sublem kernels
void time_kernels_elas(Mesh *mesh){

  double gflops = 0.0;
  double bw = 0.0;
  double denom = (double) (100 * mesh->K * p_Np * p_Nfields);
  FILE *timingFile = fopen ("blockTimings.txt","a");

  occa::initTimer(device);
  
  timeV = 0.0;
  timeS = 0.0;
  timeU = 0.0;
  /*
  // nodal kernels
  for (int step = 0; step < 100; ++step){
    dfloat fdt = 1.f, rka = 1.f, rkb = 1.f, ftime=1.f;

    occa::tic("volume_elas (bern)");
    rk_volume_bern_elas(mesh->K, c_vgeo, c_D_ids1, c_D_ids2, c_D_ids3, c_D_ids4, c_Dvals4, c_Q, c_rhsQ);
    device.finish();
    dfloat elapsedV = occa::toc("volume_elas (bern)",rk_volume_bern_elas, gflops, bw * sizeof(dfloat));

    occa::tic("surface_elas (bern))");
    rk_surface_bern_elas(mesh->K, c_fgeo, c_Fmask, c_vmapP, c_slice_ids, c_EEL_ids, c_EEL_vals, c_L0_ids, c_L0_vals, c_cEL, c_Q, c_rhsQ);
    device.finish();
    dfloat elapsedS = occa::toc("surface_elas (bern)",rk_surface_bern_elas, gflops, bw * sizeof(dfloat));

    occa::tic("update_elas_full (bern)");
    rk_update_elas(mesh->K, c_Vq_BB, c_Pq_BB, c_rhoq, c_lambdaq, c_muq, c_c11, c_c12,ftime, c_fsrc,rka, rkb, fdt,c_rhsQ, c_resQ, c_Q);
    device.finish();
    dfloat elapsedU = occa::toc("update_elas_full (nodal)",rk_update_elas, gflops, bw * sizeof(dfloat));

    timeV+=elapsedV;
    timeS+=elapsedS;
    timeU+=elapsedU;
  }
  occa::printTimer();
  printf("Nodal kernels: elapsed time per timestep: V = %g, S = %g, U = %g\n", timeV/100,timeS/100,timeU/100);
  printf("full quadrature kernel time per element: %g\n", timeU/(100*mesh->K));
  timeV /= denom;
  timeS /= denom;
  timeU /= denom;
  printf("Nodal kernels: elapsed time per dof per timestep: V = %g, S = %g, U = %g, Total = %g\n", timeV,timeS,timeU,timeV+timeS+timeU);
  fprintf(timingFile,"%%Nodal kernels for N = %d\n KblkV(%d,%d) = %4.4g; KblkS(%d,%d) = %4.4g; KblkU(%d,%d) = %4.4g;\n",
          p_N,p_N,KblkV,timeV,p_N,KblkS,timeS,p_N,KblkU,timeU);
  */
  // now do Bernstein kernels
  timeV = 0.0;
  timeS = 0.0;
  timeU = 0.0;
  for (int step = 0; step < 100; ++step){
    dfloat fdt = 1.f, rka = 1.f, rkb = 1.f, ftime=1.f;

    occa::tic("volume_elas (bern)");
    rk_volume_bern_elas(mesh->K, c_vgeo, c_D_ids1, c_D_ids2, c_D_ids3, c_D_ids4, c_Dvals4, c_Q, c_rhsQ);
    device.finish();
    dfloat elapsedV = occa::toc("volume_elas (bern)",rk_volume_bern_elas, gflops, bw * sizeof(dfloat));

    occa::tic("surface_elas (bern)");
    rk_surface_bern_elas(mesh->K, c_fgeo, c_Fmask, c_vmapP, c_slice_ids, c_EEL_ids, c_EEL_vals, c_L0_ids, c_L0_vals, c_cEL, c_Q, c_rhsQ);
    device.finish();
    dfloat elapsedS = occa::toc("surface_elas (bern)",rk_surface_bern_elas, gflops, bw * sizeof(dfloat));

    occa::tic("update_elas (bern)");
    rk_update_bern_elas(mesh->K, c_col_id, c_col_val, c_L_id, c_rho_BB, c_lambda_BB, c_mu_BB, c_ENMT_val, c_ENMT_id, c_ENM_val, c_ENM_id, c_E, c_co, c_ENMT_index, ftime, c_fsrc_BB, rka, rkb, fdt, c_rhsP, c_resP, c_P);
    device.finish();
    dfloat elapsedU = occa::toc("update_elas (bern)",rk_update_bern_elas, gflops, bw * sizeof(dfloat));

    
    timeV+=elapsedV;
    timeS+=elapsedS;
    timeU+=elapsedU;
  }
  occa::printTimer();

  printf("Bern kernels: elapsed time per timestep: V = %g, S = %g, U = %g\n", timeV/100,timeS/100, timeU/100);
  printf("BBWADG kernel time per element: %g\n", timeU/(100*mesh->K));
  timeV /= denom;
  timeS /= denom;
  timeU /= denom;
  
  printf("Bern kernels: elapsed time per dof per timestep: V = %g, S = %g, U = %g, Total = %g\n", timeV,timeS,timeU,timeV+timeS+timeU);
#if USE_SLICE_LIFT
  fprintf(timingFile,"%%Bern kernels for N = %d\n KblkVB(%d,%d) = %4.4g; KblkSB_slice(%d,%d) = %4.4g;\n", p_N,p_N,KblkV,timeV,p_N,KblkS,timeS);
#else
  fprintf(timingFile,"%%Bern kernels for N = %d\n KblkVB(%d,%d) = %4.4g; KblkSB(%d,%d) = %4.4g;\n", p_N,p_N,KblkV,timeV,p_N,KblkS,timeS);
#endif
  fclose(timingFile);

}



// run RK
void Wave_RK(Mesh *mesh, dfloat FinalTime, dfloat dt, int useWADG){

  double time = 0;
  int    INTRK, tstep=0;

  int totalSteps = (int)floor(FinalTime/dt);
  int tchunk = max(totalSteps/10,1);

  int tsample = 2*(p_N+1)*(p_N+1); // sample at every (*) timesteps
  int num_samples = totalSteps/tsample + 1;

  double *L2err = (double*) calloc(num_samples,sizeof(double));
  double *tvec = (double*) calloc(num_samples,sizeof(double));
  int tstep_sample = 0;

  /* outer time step loop  */
  while (time < FinalTime){
    if (tstep%tchunk==0){
      printf("on timestep %d/%d\n",tstep, totalSteps);
    }

    /* adjust final step to end exactly at FinalTime */
    if (time+dt > FinalTime) { dt = FinalTime-time; }
    
    for (INTRK=1; INTRK<=5; ++INTRK) {
      
      // compute DG rhs
      const dfloat fdt = dt;
      const dfloat fa = (float)mesh->rk4a[INTRK-1];
      const dfloat fb = (float)mesh->rk4b[INTRK-1];
      
      RK_step_WADG_subelem(mesh, fa, fb, fdt, time);
    }
    
    time += dt;     /* increment current time */
    tstep++;        /* increment timestep */

  }

}

// defaults to nodal!!
void RK_step_WADG_subelem(Mesh *mesh, dfloat rka, dfloat rkb, dfloat fdt, dfloat time){

  dfloat f0 = 10.0;
  dfloat tR = 1.0 / f0;
  dfloat at = M_PI*f0*(time-tR);
  dfloat ftime = 1e4*(1.0 - 2.0*at*at)*exp(-at*at); // ricker pulse
  // kernels for Bernstein
  rk_volume_bern_elas(mesh->K, c_vgeo, c_D_ids1, c_D_ids2, c_D_ids3, c_D_ids4, c_Dvals4, c_Q, c_rhsQ);
  rk_surface_bern_elas(mesh->K, c_fgeo, c_Fmask, c_vmapP, c_slice_ids, c_EEL_ids, c_EEL_vals, c_L0_ids, c_L0_vals, c_cEL, c_Q, c_rhsQ);
  rk_update_bern_elas(mesh->K, c_col_id, c_col_val, c_L_id, c_rho_BB, c_lambda_BB, c_mu_BB, c_ENMT_val, c_ENMT_id, c_ENM_val, c_ENM_id, c_E, c_co, c_ENMT_index, ftime, c_fsrc_BB, rka, rkb, fdt, c_rhsQ, c_resQ, c_Q);
  
  // kernels for Nodal
  rk_volume_bern_elas(mesh->K, c_vgeo, c_D_ids1, c_D_ids2, c_D_ids3, c_D_ids4, c_Dvals4, c_P, c_rhsP);
  rk_surface_bern_elas(mesh->K, c_fgeo, c_Fmask, c_vmapP, c_slice_ids, c_EEL_ids, c_EEL_vals, c_L0_ids, c_L0_vals, c_cEL, c_P, c_rhsP);
  //rk_update_bern_elas(mesh->K, c_col_id, c_col_val, c_L_id, c_rho_BB, c_lambda_BB, c_mu_BB, c_ENMT_val, c_ENMT_id, c_ENM_val, c_ENM_id, c_E, c_co, c_ENMT_index, ftime, c_fsrc_BB, rka, rkb, fdt, c_rhsP, c_resP, c_P);
  rk_update_elas(mesh->K, c_Vq_BB, c_Pq_BB, c_rhoq, c_lambdaq, c_muq, c_c11, c_c12,ftime, c_fsrc,rka, rkb, fdt,c_rhsP, c_resP, c_P);

  device.finish();

}

void RK_step(Mesh *mesh, dfloat rka, dfloat rkb, dfloat fdt){

  double gflops = 0.0;
  double bw = 0.0;
  int K = mesh->K;

#if USE_BERN
  //printf("using bernstein kernels\n");
  rk_volume_bern(mesh->K, c_vgeo,
		 c_D_ids1, c_D_ids2, c_D_ids3, c_D_ids4, c_Dvals4,
		 c_Q, c_rhsQ);

  rk_surface_bern(mesh->K, c_fgeo, c_Fmask, c_vmapP,
		  c_slice_ids,
		  c_EEL_ids, c_EEL_vals,
   		  c_L0_ids, c_L0_vals,
		  c_cEL,
      		  c_Q, c_rhsQ);
#else
  //printf("Using nodal kernels\n");
  rk_volume(mesh->K, c_vgeo, c_Dr, c_Ds, c_Dt, c_Q, c_rhsQ);

  rk_surface(mesh->K, c_fgeo, c_Fmask, c_vmapP, c_LIFT, c_Q, c_rhsQ);

#endif

  rk_update(mesh->K, rka, rkb, fdt, c_rhsQ, c_resQ, c_Q);
  //int Ntotal = p_Nfields*p_Np*mesh->K;
  //rk_update(Ntotal, rka, rkb, fdt, c_rhsQ, c_resQ, c_Q);

#if 1
  dfloat *f_Q = (dfloat*) calloc(p_Nfields*mesh->K*p_Np, sizeof(dfloat));
  c_Q.copyTo(f_Q);
  for(int fld = 0; fld < p_Nfields; ++fld){
    printf("Field solution %d: \n",fld);
    for(int i = 0; i < p_Np; ++i){
      //      for(int e = 0; e < mesh->K; ++e){
      for(int e = 0; e < 1; ++e){
	printf("%f ",f_Q[i + fld*p_Np + e*p_Nfields*p_Np]);
      }
      printf("\n");
    }
    printf("\n\n");
  }
#endif

}




// set initial condition
void WaveSetU0(Mesh *mesh, dfloat *Q, dfloat time, int field,
	       double(*uexptr)(double,double,double,double)){

  // write out field = fields 2-4 = 0 (velocity)
  for(int k = 0; k < mesh->K; ++k){

    // store local interpolant
    dfloat *Qloc = BuildVector(p_Np);
    for(int i = 0; i < p_Np; ++i){
      double x = mesh->x(i,k);
      double y = mesh->y(i,k);
      double z = mesh->z(i,k);
      Qloc[i] = (*uexptr)(x,y,z,0.0);
    }

#if 0 // convert nodal values to bernstein coefficients

    dfloat *Qtmp = BuildVector(p_Np);
    for (int i = 0; i < p_Np; ++i){
      Qtmp[i] = 0.f;
      for (int j = 0; j < p_Np; ++j){
	Qtmp[i] += mesh->invVB(i,j)*Qloc[j];
      }
    }
    for (int i = 0; i < p_Np; ++i){
      Qloc[i] = Qtmp[i];
    }

#endif

    for (int i = 0; i < p_Np; ++i){
      int id = k*p_Np*p_Nfields + i + p_Np*field;
      Q[id] = Qloc[i];
    }
  }
}


// set initial condition
void WaveSetU0P0(Mesh *mesh, dfloat *Q, dfloat *P, dfloat time, int field,
		 double(*uexptr)(double,double,double,double)){

  // write out field = fields 2-4 = 0 (velocity)
  for(int k = 0; k < mesh->K; ++k){

    // store local interpolant
    dfloat *Qloc = BuildVector(p_Np);
    dfloat *Ploc = BuildVector(p_Np);
    for(int i = 0; i < p_Np; ++i){
      double x = mesh->x(i,k);
      double y = mesh->y(i,k);
      double z = mesh->z(i,k);
      Qloc[i] = (*uexptr)(x,y,z,0.0);
      Ploc[i] = Qloc[i];
    }

    // convert nodal values to bernstein coefficients
    
    dfloat *Qtmp = BuildVector(p_Np);
    
    for (int i = 0; i < p_Np; ++i){
      Qtmp[i] = 0.f;
      
      for (int j = 0; j < p_Np; ++j){
	Qtmp[i] += mesh->invVB(i,j)*Qloc[j];
      }
    }
    for (int i = 0; i < p_Np; ++i){
      Qloc[i] = Qtmp[i];
    }

    
    // Convert both P and Q to bernstein coefficients
    for (int i = 0; i < p_Np; ++i){
      int id = k*p_Np*p_Nfields + i + p_Np*field;
      Q[id] = Qloc[i];
      P[id] = Qloc[i];
    }
  }
}


// set initial condition
void WaveProjectU0(Mesh *mesh, dfloat *Q, dfloat time,int field,
		   double(*uexptr)(double,double,double,double)){

  int Nq = mesh->Nq;
  VectorXd wq = mesh->wq;
  MatrixXd Vq = mesh->Vq;
  printf("Nq = %d, size of xq = %d\n",Nq,mesh->xq.rows());
  printf("size of Vq = %d, %d\n",Vq.rows(),Vq.cols());

  MatrixXd Qloc(p_Np,1);

  // write out field = fields 2-4 = 0 (velocity)
  for(int k = 0; k < mesh->K; ++k){

    // compute mass matrix explicitly w/quadrature
    VectorXd Jq = mesh->Jq.col(k);
    MatrixXd Mloc = Vq.transpose() * wq.asDiagonal() * Jq.asDiagonal() * Vq;

    // compute fxn at quad nodes
    MatrixXd uq(Nq,1);
    for (int i = 0; i < Nq; ++i){
      double xq = mesh->xq(i,k);
      double yq = mesh->yq(i,k);
      double zq = mesh->zq(i,k);
      double uqi = (*uexptr)(xq,yq,zq,0.0);
      //printf("uq[%d] = %g\n",i,uqi);
      uq(i,0) = uqi*wq(i)*Jq(i);
    }
    MatrixXd b = mesh->Vq.transpose() * uq;
    Qloc = mldivide(Mloc,b);

#if USE_BERN // convert nodal values to bernstein coefficients

    Qloc =  mesh->invVB*Qloc;

#endif

    // set pressure
    for (int i = 0; i < p_Np; ++i){
      int id = k*p_Np*p_Nfields + i + field*p_Np;
      Q[id] = Qloc(i,0);
    }

    //cout << "Q on elem " << k << " initialized to " << endl << Qloc << endl;

  }


  //cout << "Vq = " << endl << mesh->Vq << endl;
  //cout << "Jq = " << endl << mesh->Jq << endl;
  //cout << "wq = " << endl << mesh->wq << endl;
  //cout << "xq = " << endl << mesh->xq << endl;
  //cout << "yq = " << endl << mesh->yq << endl;
  //cout << "zq = " << endl << mesh->zq << endl;

}


void WaveSetData3d(dfloat *Q, dfloat *P){
  c_Q.copyFrom(Q);
  c_P.copyFrom(P);
}


void WaveGetData3d(Mesh *mesh, dfloat *Q, dfloat *P){
  c_Q.copyTo(Q);
  c_P.copyTo(P);

 
  dfloat *Qtmp = BuildVector(p_Np);
  dfloat *Ptmp = BuildVector(p_Np);
  for (int fld = 0; fld < p_Nfields; fld++){
    for (int e = 0; e < mesh->K; ++e){
      for (int i = 0; i < p_Np; ++i){
	Qtmp[i] = 0.f;
	Ptmp[i] = 0.f;
	for (int j = 0; j < p_Np; ++j){
	  int id = j + fld*p_Np + e*p_Nfields*p_Np;
	  Qtmp[i] += mesh->VB(i,j)*Q[id];
	  Ptmp[i] += mesh->VB(i,j)*P[id];
	}
      }
      for (int i = 0; i < p_Np; ++i){
	int id = i + fld*p_Np + e*p_Nfields*p_Np;
	Q[id] = Qtmp[i];
	P[id] = Ptmp[i];
      }
    }
  }
}

void writeVisToGMSH(string fileName, Mesh *mesh,dfloat *Q, int iField, int Nfields){

  int timeStep = 0;
  double time = 0.0;
  int K = mesh->K;
  int Dim = 3;
  int N = p_N;


  MatrixXi monom(p_Np, Dim);
  MatrixXd vdm(p_Np, p_Np);
  for(int i=0, n=0; i<=N; i++){
    for(int j=0; j<=N; j++){
      for(int k=0; k<=N; k++){
	if(i+j+k <= N){
	  monom(n,0) = i;
	  monom(n,1) = j;
	  monom(n,2) = k;
	  n++;
	}
      }
    }
  }
  for(int m=0; m<p_Np; m++){
    for(int n=0; n<p_Np; n++){
      double r = mesh->r(n);
      double s = mesh->s(n);
      double t = mesh->t(n);
      vdm(m,n) = pow((r+1)/2.,monom(m,0)) *
	pow((s+1)/2.,monom(m,1)) * pow((t+1)/2.,monom(m,2));
    }
  }
  MatrixXd coeff = vdm.inverse();

  /// write the gmsh file
  ofstream *posFile;
  posFile = new ofstream(fileName.c_str());
  *posFile << "$MeshFormat" << endl;
  *posFile << "2.2 0 8" << endl;
  *posFile << "$EndMeshFormat" << endl;

  /// write the interpolation scheme
  *posFile << "$InterpolationScheme" << endl;
  *posFile << "\"MyInterpScheme\"" << endl;
  *posFile << "1" << endl;
  *posFile << "5 2" << endl;  // 5 2 = tets
  *posFile << p_Np << " " << p_Np << endl;  // size of matrix 'coeff'
  for(int m=0; m<p_Np; m++){
    for(int n=0; n<p_Np; n++)
      *posFile << coeff(m,n) << " ";
    *posFile << endl;
  }
  *posFile << p_Np << " " << Dim << endl;  // size of matrix 'monom'
  for(int n=0; n<p_Np; n++){
    for(int d=0; d<Dim; d++)
      *posFile << monom(n,d) << " ";
    *posFile << endl;
  }
  *posFile << "$EndInterpolationScheme" << endl;

  /// write element node data
  *posFile << "$ElementNodeData" << endl;
  *posFile << "2" << endl;
  *posFile << "\"" << "Field " << iField << "\"" << endl;  /// name of the view
  *posFile << "\"MyInterpScheme\"" << endl;
  *posFile << "1" << endl;
  *posFile << time << endl;
  *posFile << "3" << endl;
  *posFile << timeStep << endl;
  *posFile << "1" << endl;  /// ("numComp")
  *posFile << K << endl;  /// total number of elementNodeData in this file
  for(int k=0; k<K; k++){
    *posFile << mesh->EToGmshE(k) << " " << p_Np;
    for(int i=0; i<p_Np; i++)
      *posFile << " " << Q[i + iField*p_Np + k*p_Np*Nfields];
    *posFile << endl;
  }
  *posFile << "$EndElementNodeData" << endl;

  posFile->close();
  delete posFile;

}



void compute_error_adaptive(Mesh *mesh, dfloat *Q, dfloat *P, double &L2error, double &reL2error){

  //compute error                                                                                           
  L2error = 0.0;
  double L2norm = 0.0;
  for(int k = 0; k < mesh->K; ++k){
    for(int i = 0; i < mesh->Nq; ++i){
      
      double J = mesh->J(0,k);
      double uq = 0.0;
      double up = 0.0;
      
      for(int j=0; j<p_Np;++j){
        double Vq = mesh->Vq(i,j);
        uq += Vq * Q[j+p_Np*p_Nfields*k];
        up += Vq * P[j+p_Np*p_Nfields*k];
      }
      
      double err = uq - up;
      
      L2error += err*err*mesh->wq(i)*J;
      L2norm  += uq*uq*mesh->wq(i)*J;
    }
  }
  L2error = sqrt(L2error);
  reL2error=L2error/sqrt(L2norm);
  
  return;
}
