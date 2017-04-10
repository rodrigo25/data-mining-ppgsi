load kmeansdata
k=4;

[U, kpoints] = kmeans(X,k,'e');

k1 = U(1,:)==1;
k2 = U(2,:)==1;
k3 = U(3,:)==1;
k4 = U(4,:)==1;


figure

subplot(2,2,1)
scatter3(X(k1,1), X(k1,2), X(k1,3), 'ro')
hold on
scatter3(X(k2,1), X(k2,2), X(k2,3), 'bx')
scatter3(X(k3,1), X(k3,2), X(k3,3), 'gd')
scatter3(X(k4,1), X(k4,2), X(k4,3), 'ms')
scatter3(kpoints(:,1),kpoints(:,2),kpoints(:,3),40,'k','filled')

subplot(2,2,2)
scatter3(X(k1,1), X(k1,2), X(k1,4), 'ro')
hold on
scatter3(X(k2,1), X(k2,2), X(k2,4), 'bx')
scatter3(X(k3,1), X(k3,2), X(k3,4), 'gd')
scatter3(X(k4,1), X(k4,2), X(k4,4), 'ms')
scatter3(kpoints(:,1),kpoints(:,2),kpoints(:,4),40,'k','filled')

subplot(2,2,3)
scatter3(X(k1,1), X(k1,3), X(k1,4), 'ro')
hold on
scatter3(X(k2,1), X(k2,3), X(k2,4), 'bx')
scatter3(X(k3,1), X(k3,3), X(k3,4), 'gd')
scatter3(X(k4,1), X(k4,3), X(k4,4), 'ms')
scatter3(kpoints(:,1),kpoints(:,3),kpoints(:,4),40,'k','filled')

subplot(2,2,4)
scatter3(X(k1,2), X(k1,3), X(k1,4), 'ro')
hold on
scatter3(X(k2,2), X(k2,3), X(k2,4), 'bx')
scatter3(X(k3,2), X(k3,3), X(k3,4), 'gd')
scatter3(X(k4,2), X(k4,3), X(k4,4), 'ms')
scatter3(kpoints(:,2),kpoints(:,3),kpoints(:,4),40,'k','filled')
