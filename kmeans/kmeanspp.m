function [ U, kpoints ] = kmeanspp( data, k, dist )
           
  [n, m] = size(data);
  K_ini = zeros(k,1);
  kpoints = zeros(k,m);
  
  %SELE��O DE CENTROIDES INICIAIS
  
  %Seleciona primeiro centroide inicial
  K_ini(1) = randi(n,1);
  kpoints(1,:) = data(K_ini(1),:);
  
  %Seleciona outros centroides iniciais
  for i=2:k
    Distancias = Inf(i-1,n);
    for p=1:n
      Distancias(:,p) = distance(data(p,:), kpoints(1:i-1,:), i-1, dist);
    end
    D = min(Distancias,[],1); %Define distancia dos centroides mais pr�ximos
    %D(K_ini>0)=0; %elimina centroides j� selecionados
    prob = (D.^2)./sum(D.^2);
    
    K_ini(i) = sum(rand >= cumsum([0, prob]));
    kpoints(i,:) = data(K_ini(i),:);
  end
  
  
  
  
  %K-MEANS NORMAL
  
  % DEFINI��O DE VARI�VEIS
  maxIt = 100; % maximo de itera��es
  it = 0; %contador de itera��es
  U = zeros(k, n); %cria matriz de parti��o
  old_kpoints = kpoints+5; %prot�tipos anteriores para controle
  epsilon = 0; %erro m�ximo
  
  % LA�O DE ITERA��ES DO K-MEANS
  while(abs(old_kpoints - kpoints)>epsilon & (it<maxIt))
    it = it+1;
    old_kpoints = kpoints;
    
    % ATUALIZA MATRIZ DE PARTI��O
    for i=1:n   %para cada instancia
      p = data(i,:);
      D = distance(p, kpoints, k, dist); %calcula distancias dos prot�tipos
      [~,ind] = min(D); %Verifica qual o prot�tipo de menor distancia
      U(:,i)=0; %Zera todos os prot�tipos desse ponto
      U(ind,i)=1; %atribui 1 ao prot�tipo de menor distancia
    end
          
    % PLOTA GR�FICO (Para problemas bidimensionais)        
    %plotClusters2D(data,kpoints,U,k,[]);
    %return
    %pause;
    
    % ATUALIZA PROT�TIPOS
    for i=1:k  %para cada prot�tipo
      kpoints(i,:) = (U(i,:)*data)/length(find(U(i,:)==1)); %atualiza com a m�dia
    end

  end
  
  
  % �LTIMA ATUALIZA��O DA MATRIZ
  for i=1:n   %para cada instancia
    p = data(i,:);
    D = distance(p, kpoints, k, dist); %calcula distancias dos prot�tipos
    [~,ind] = min(D); %Verifica qual o prot�tipo de menor distancia
    U(:,i)=0; %Zera todos os prot�tipos desse ponto
    U(ind,i)=1; %atribui 1 ao prot�tipo de menor distancia
  end
  
  %fprintf('n�mero de itera��es: %i\n\n', it)
  
  
  
  
end


% FUN��O DE DIST�NCIAS
function D = distance(p, data, n, dist)
  P = repmat(p,n,1);
  if strcmp(dist,'e') %distancia euclidiana
    D = sqrt(sum((P-data).^2, 2));
  elseif strcmp(dist,'m') %distancia manhattan
    D = sum(abs(P-data), 2);
  end
end

