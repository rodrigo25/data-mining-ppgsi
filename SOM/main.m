load kmeansdata
k=4;

[n,m] = size(X);

X = (X-repmat(mean(X),n,1))./repmat(std(X),n,1); % Z-SCORE NORALIZATION

Nx = 10;
Ny = 10;
Ns = Nx*Ny;

[W] = SOM( X, [Nx Ny], 'gauss');


grid = (Nx);

for i=1:Ns
  t = 0;
  soma = 0;
  if((fix((i-1)/Nx) ~= 0))%linha pra cima
    t = t+1;
    norm(W(i)-W(i-Nx))
    plot([W(i,1) W(i-Nx,1)],[W(i,2) W(i-Nx,2)],'r')
  end
  if (rem(i-1,Nx) ~= Nx-1)%linha pra direita
    t = t+1;
    %T(i,i+1,a)
    plot([W(i,1) W(i+1,1)],[W(i,2) W(i+1,2)],'r')
  end
  if (fix((i-1)/Nx) ~= Nx-1)%linha pra baixo
    t = t+1;
    %T(i,i+n,a)
    plot([W(i,1) W(i+Nx,1)],[W(i,2) W(i+Nx,2)],'r')
  end
  if (rem(i-1,Nx) ~= 0)%linha pra esquerda
    t = t+1;
    %T(i,i-1,a)
    plot([W(i,1) W(i-1,1)],[W(i,2) W(i-1,2)],'r')
  end
  
  /t;
end



%[U, kpoints] = kmeans(X,k,'e'); % K-means
scatter(W(:,1), W(:,2))
hold on
for i=1:Ns
  if((fix((i-1)/Nx) ~= 0))%linha pra cima
    %T(i,i-n,a)
    plot([W(i,1) W(i-Nx,1)],[W(i,2) W(i-Nx,2)],'r')
  end
  if (rem(i-1,Nx) ~= Nx-1)%linha pra direita
    %T(i,i+1,a)
    plot([W(i,1) W(i+1,1)],[W(i,2) W(i+1,2)],'r')
  end
  if (fix((i-1)/Nx) ~= Nx-1)%linha pra baixo
    %T(i,i+n,a)
    plot([W(i,1) W(i+Nx,1)],[W(i,2) W(i+Nx,2)],'r')
  end
  if (rem(i-1,Nx) ~= 0)%linha pra esquerda
    %T(i,i-1,a)
    plot([W(i,1) W(i-1,1)],[W(i,2) W(i-1,2)],'r')
  end
end

%neuronsGrid = [ceil((1:Ns)/dim(1));mod((0:Ns-1),dim(1))+1]';

%plotClusters2D(X,kpoints,U,k,{'X','Y','Z','T'});