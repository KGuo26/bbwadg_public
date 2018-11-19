//Mesh.CharacteristicLengthMax = 1;
//Mesh.CharacteristicLengthFactor = .2;
Mesh.Optimize = 1;
h = 0.0325;

Point(1) = {-1.0,-1.0,0.0,h};
Point(2) = {1.0,-1.0,0.0,h};
Point(3) = {1.0,1.0,0.0,h};
Point(4) = {-1.0,1.0,0.0,h};
Line(1) = {4,3};
Line(2) = {3,2};
Line(3) = {2,1};
Line(4) = {1,4};
Line Loop(5) = {2,3,4,1};
Plane Surface(6) = {5};
Extrude {0.0,0.0,1}{
  Surface{6};
}

Extrude {0.0,0.0,-1.0} {
  Surface{6};
}
