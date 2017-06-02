
r = .2 + (.3-.2).*rand(100,1);
r2 = .2 + (.3-.2).*rand(100,1);
r3 = .2 + (.3-.2).*rand(100,1);

b = .1 + (.2-.1).*rand(100,1);
b2 = .8 + (.9-.8).*rand(100,1);
b3 = .8 + (.9-.8).*rand(100,1);

c = .8 + (.9-.8).*rand(100,1);
c2 = .4 + (.5-.4).*rand(100,1);
c3 = .7 + (.8-.7).*rand(100,1);

X = [r r2 r3; b b2 b3; c c2 c3];

%X = (X-repmat(mean(X),300,1))./repmat(std(X),300,1); % Z-SCORE NORALIZATION

%scatter(X(:,1), X(:,2))

%return

plot3(X(:,1), X(:,2),X(:,3), 'x')
hold on
set(gca,'XLim',[0 1])
hold on
set(gca,'YLim',[0 1])

%save data_Test1 X
