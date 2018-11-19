#include <stdio.h>
#include "fem.h"

#include <occa.hpp>

bool edgeCompare(vector<unsigned int> a, vector<unsigned int> b){
  if (a[0]==b[0]){
    return a[1] < b[1];
  }
  return a[0] < b[0];
}
bool faceCompare(vector<unsigned int> a, vector<unsigned int> b){
  if (a[0]==b[0]){
    if (a[1]==b[1]){
      return a[2] < b[2];
    }
    return a[1] < b[1];
  }
  return a[0] < b[0];
}

// set up CG data structures - could be useful in future?
void setupCG(Mesh *mesh){

  unsigned int K = mesh->K;

  // total number of vertices, edges, faces
  unsigned int Nv = 0;
  for (int e = 0; e < K; ++e){
    for (int v = 0; v < 4; ++v){
      Nv = max(Nv,mesh->EToV[e][v]);
      //printf("EToV(%d,%d) = %d\n",e,v,mesh->EToV[e][v]);
    }
  }
  Nv += 1; // for zero indexing

  // compute number of unique faces/edges
  int Nedges = 6;
  const int edgenum[6][2] = { {0,1}, {1,2}, {2,0}, {0,3}, {1,3}, {2,3} };

  // face,vertex id
  const int vnum[4][3] = { {0,1,2}, {0,1,3}, {1,2,3}, {0,2,3} };

  vector<vector<unsigned int> > edges;
  vector<vector<unsigned int> > faces;
  for (unsigned int e = 0; e < K; ++e){
    for (int edge = 0; edge < Nedges; ++edge){
      vector<unsigned int> edgeVerts;
      edgeVerts.push_back(mesh->EToV[e][edgenum[edge][0]]);
      edgeVerts.push_back(mesh->EToV[e][edgenum[edge][1]]);
      std::sort(edgeVerts.begin(),edgeVerts.end()); // sort vertices

      // tag elem, edge
      edgeVerts.push_back(e);
      edgeVerts.push_back(edge);
      edges.push_back(edgeVerts);
    }

    for (int f = 0; f < p_Nfaces; ++f){
      vector<unsigned int> faceVerts;
      faceVerts.push_back(mesh->EToV[e][vnum[f][0]]);
      faceVerts.push_back(mesh->EToV[e][vnum[f][1]]);
      faceVerts.push_back(mesh->EToV[e][vnum[f][2]]);
      std::sort(faceVerts.begin(),faceVerts.end()); // sort vertices

      // tag elem, face
      faceVerts.push_back(e);
      faceVerts.push_back(f);
      faces.push_back(faceVerts);
    }
  }
  // sort face/edge tuples
  std::sort(edges.begin(),edges.end(),edgeCompare);
  std::sort(faces.begin(),faces.end(),faceCompare);

  // assign unique ids to edges/faces
  int edge_id = 0;
  for (int i = 0; i < edges.size()-1; ++i){
    edges[i].push_back(edge_id);
    if (edgeCompare(edges[i],edges[i+1])){ // if f_i != f_{i+1}
      ++edge_id;
    }
  }
  edges[edges.size()-1].push_back(edge_id);
  int NedgesK = edge_id+1;

  int face_id = 0;
  for (int i = 0; i < faces.size()-1; ++i){
    faces[i].push_back(face_id);
    if (faceCompare(faces[i],faces[i+1])){ // if f_i != f_{i+1}
      ++face_id;
    }
  }
  faces[faces.size()-1].push_back(face_id);
  int NfacesK = face_id+1;

  //  printf("Total edges = %d, total faces = %d\n",NedgesK,NfacesK);
#if 0
  for (int i = 0; i < edges.size(); ++i){
    printf("edges %d = [%d, %d]. edge id = %d\n",
	   i,edges[i][0],edges[i][1],edges[i][4]);
  }

  for (int i = 0; i < faces.size(); ++i){
    printf("face %d = [%d, %d, %d]. face id = %d\n",
	   i,faces[i][0],faces[i][1],faces[i][2],faces[i][5]);
  }
#endif

  MatrixXi EToEdgeK(K,Nedges);
  for (int i = 0; i < edges.size(); ++i){
    int e = edges[i][2];
    int edge = edges[i][3];
    int id = edges[i][4];
    EToEdgeK(e,edge) = id;
  }
  MatrixXi EToFaceK(K,p_Nfaces);
  for (int i = 0; i < faces.size(); ++i){
    int e = faces[i][3];
    int face = faces[i][4];
    int id = faces[i][5];
    EToFaceK(e,face) = id;
  }

#if 0
  cout << "EToEdgeK = " << endl << EToEdgeK << endl;
  cout << "EToFaceK = " << endl << EToFaceK << endl;
#endif

  // compute number of global dofs
  int NpEdge = max(0,p_N-1); // interior edge dofs
  int NpFace = max(0,(p_N-1)*(p_N-2)/2); // interior face dofs
  int NpInt  = max(0,(p_N-1)*(p_N-2)*(p_N-3)/6); // interior volume dofs

  //printf("For N = %d, dofs per edge = %d, per face = %d, per interior = %d\n",
  //	 p_N,NpEdge,NpFace,NpInt);

  int NpTri = (p_N+1)*(p_N+2)/2;
  VectorXi vert_ids(4); vert_ids << 0, p_N, NpTri-1, p_Np-1;
  MatrixXi edge_vert_ids(p_N+1,Nedges); edge_vert_ids.fill(0);
  MatrixXi face_ids(NpFace,p_Nfaces); face_ids.fill(0);
  VectorXi int_ids(NpInt); int_ids.fill(0);

  // save ids
  int sk = 0;
  int esk1 = 0,esk2 = 0, esk3 = 0, esk4 = 0, esk5 = 0, esk6 = 0;
  int fsk1 = 0,fsk2 = 0, fsk3 = 0, fsk4 = 0;
  int isk = 0;
  for (int k = 0; k <= p_N; ++k){
    for (int j = 0; j <= p_N-k; ++j){
      for (int i = 0; i <= p_N-j-k; ++i){
	// vertex ids computed explicitly

	// save edge ids (with extra vertex nodes)
	if (k==0 && j==0){
	  edge_vert_ids(esk1,0) = sk; ++esk1;
	}
	if ((i+j)==p_N){
	  edge_vert_ids(esk2,1) = sk; ++esk2;
	}
	if (k==0 && i==0){
	  edge_vert_ids(esk3,2) = sk; ++esk3;
	}
	if (i==0 && j==0){
	  edge_vert_ids(esk4,3) = sk; ++esk4;
	}
	if ((i+k)==p_N){
	  edge_vert_ids(esk5,4) = sk; ++esk5;
	}
	if ((j+k)==p_N){
	  edge_vert_ids(esk6,5) = sk; ++esk6;
	}

	// save interior face ids
	if (k==0 && i > 0 && j > 0 && i+j<p_N){
	  face_ids(fsk1,0) = sk; ++fsk1;
	}
	if (j==0 && i > 0 && k > 0 && i+k<p_N){
	  face_ids(fsk2,1) = sk; ++fsk2;
	}
	if (i+j+k==p_N && i > 0 && j > 0 && i+j<p_N){
	  face_ids(fsk3,2) = sk; ++fsk3;
	}
	if (i==0 && k > 0 && j > 0 && j+k < p_N){
	  face_ids(fsk4,3) = sk; ++fsk4;
	}
	// compute volume ids
	int interior =
	  (i > 0) && (i < p_N) &&
	  (j > 0) && (j < p_N) &&
	  (k > 0) && (k < p_N) &&
	  (i+j+k < p_N);
	if (interior){
	  int_ids(isk) = sk; ++isk;
	}
	++sk;

      }
    }
  }
  MatrixXi edge_ids(NpEdge,Nedges); edge_ids.fill(0);
  for (int i = 0; i < NpEdge;++i){
    edge_ids.row(i) = edge_vert_ids.row(i+1);
  }
  //cout << "sk = " << sk << endl;
  //cout << "vert_ids = " << endl << vert_ids << endl;
  //cout << "edge_ids = " << endl << edge_ids << endl;// extra rows
  //cout << "face_ids = " << endl << face_ids << endl;
  //cout << "int_ids = " << endl << int_ids << endl;

  MatrixXi localToGlobalNode(p_Np,K);
  for (int e = 0; e < K; ++e){
    for (int vi = 0; vi < 4; ++vi){
      localToGlobalNode(vert_ids(vi),e) = mesh->EToV[e][vi];
    }
    for (int edge = 0; edge < Nedges; ++edge){
      int edge_offset = EToEdgeK(e,edge) * NpEdge + Nv;
      for (int ei = 0; ei < NpEdge; ++ei){
	localToGlobalNode(edge_ids(ei,edge),e) = ei + edge_offset;
      }
    }
    for (int face = 0; face < p_Nfaces; ++face){
      int face_offset = EToFaceK(e,face) * NpFace + NpEdge * NedgesK + Nv;
      for (int fi = 0; fi < NpFace; ++fi){
	localToGlobalNode(face_ids(fi,face),e) = fi + face_offset;
      }
    }
    int ioffset = e * NpInt + NpFace*NfacesK + NpEdge*NedgesK + Nv;
    for (int ii = 0; ii < NpInt; ++ii){
      localToGlobalNode(int_ids(ii),e) = ioffset + ii;
    }
  }
  //cout << localToGlobalNode << endl;

  int totalDofs = Nv + NpEdge*NedgesK + NpFace*NfacesK + NpInt * K;
  //  printf("Num verts = %d, num edge nodes = %d, num face nodes = %d, num interior nodes %d\n",
  //Nv, NpEdge*NedgesK,NpFace*NfacesK,NpInt*K);
  printf("Max dofs for CG = %d, %d.\n",localToGlobalNode.maxCoeff()+1,totalDofs);

  mesh->numGlobalNodes = localToGlobalNode.maxCoeff() + 1;

  // final step: correct permutation errors
  VectorXd xg(mesh->numGlobalNodes);
  VectorXd yg(mesh->numGlobalNodes);
  VectorXd zg(mesh->numGlobalNodes);
  // create globally continuous nodes
  for (int e = 0; e < mesh->K; ++e){
    for(int i = 0; i < p_Np; ++i){
      int gid = localToGlobalNode(i,e);
      xg(gid) = mesh->x(i,e);
      yg(gid) = mesh->y(i,e);
      zg(gid) = mesh->z(i,e);
    }
  }
  for (int e = 0; e < mesh->K; ++e){
    // vertex nodes should be correct already

    // match edge nodes
    VectorXi matched_eid(Nedges*NpEdge);
    for (int i = 0; i < Nedges*NpEdge; ++i){
      int gidi = localToGlobalNode(edge_ids(i),e);
      for (int j = 0; j < Nedges*NpEdge; ++j){
	int gid = localToGlobalNode(edge_ids(j),e);
	double dx = mesh->x(edge_ids(i),e) - xg(gid);
	double dy = mesh->y(edge_ids(i),e) - yg(gid);
	double dz = mesh->z(edge_ids(i),e) - zg(gid);
	double d = sqrt(dx*dx + dy*dy + dz*dz);
	if (d < NODETOL){
	  matched_eid(i) = gid;
	}
      }
    }
    for (int i = 0; i < Nedges*NpEdge;++i){
      localToGlobalNode(edge_ids(i),e) = matched_eid(i);
    }

    VectorXi matched_fid(p_Nfaces*NpFace);
    for (int i = 0; i < p_Nfaces*NpFace; ++i){
      int gidi = localToGlobalNode(face_ids(i),e);
      for (int j = 0; j < p_Nfaces*NpFace; ++j){
	int gid = localToGlobalNode(face_ids(j),e);
	double dx = mesh->x(face_ids(i),e) - xg(gid);
	double dy = mesh->y(face_ids(i),e) - yg(gid);
	double dz = mesh->z(face_ids(i),e) - zg(gid);
	double d = sqrt(dx*dx + dy*dy + dz*dz);
	if (d < NODETOL){
	  matched_eid(i) = gid;
	}
      }
    }
    for (int i = 0; i < p_Nfaces*NpFace;++i){
      localToGlobalNode(face_ids(i),e) = matched_eid(i);
    }

    // interior nodes should be fine - no permutation needed
  }

  // do final check
  for (int e = 0; e < mesh->K; ++e){
    for (int i = 0; i < p_Np; ++i){
      int gid = localToGlobalNode(i,e);
      double dx = xg(gid) - mesh->x(i,e);
      double dy = yg(gid) - mesh->y(i,e);
      double dz = zg(gid) - mesh->z(i,e);
      double d = sqrt(dx*dx + dy*dy + dz*dz);
      if (d > NODETOL){
	printf("WARNING: for elem %d, node %d, diff b/w local/global node = %g\n",e,i,d);
      }
    }
  }

  mesh->localToGlobalNodeMap = localToGlobalNode;
}
