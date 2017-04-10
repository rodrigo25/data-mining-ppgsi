load dataTest1
k=3;

[U, kpoints] = kmeans(points,k);


clf
plot(points(U(1,:)==1,1), points(U(1,:)==1,2), 'ro')
hold on
plot(points(U(2,:)==1,1), points(U(2,:)==1,2), 'bx')
plot(points(U(3,:)==1,1), points(U(3,:)==1,2), 'gd')
set(gca,'XLim',[0 1])
set(gca,'YLim',[0 1])

scatter(kpoints(:,1),kpoints(:,2),40,'k','filled')