
r = .2 + (.4-.2).*rand(100,1);
r2 = .2 + (.4-.2).*rand(100,1);

b = .3 + (.6-.3).*rand(100,1);
b2 = .7 + (.9-.7).*rand(100,1);

c = .5 + (.8-.5).*rand(100,1);
c2 = .4 + (.6-.4).*rand(100,1);

X = [r r2; b b2; c c2];

plot(X(:,1), X(:,2), 'x')
hold on
set(gca,'XLim',[0 1])
hold on
set(gca,'YLim',[0 1])

save data_Test1 X
