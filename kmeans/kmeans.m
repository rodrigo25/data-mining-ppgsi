function [ U, kpoints ] = kmeans( data, k, dist )

  [n, m] = size(data);

  % INICIA PROT�TIPOS RANDOMICAMENTE
  maximo = max(data);   %valores m�ximos de cada variavel
  minimo = min(data);   %valores minimos de cada variavel
  kpoints = zeros(k,m);
  for i=1:m     % cria k valores para a variavel m no intervalo [min max] de m
    kpoints(:,i) = minimo(i)+(maximo(i)-minimo(i)).*rand(k,1);
  end

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
  
  fprintf('n�mero de itera��es: %i\n\n', it)
  
  
end


% FUN��O DE DIST�NCIAS
function D = distance(p, kpoints, k, dist)
  P = repmat(p,k,1);
  if strcmp(dist,'e') %distancia euclidiana
    D = sqrt(sum((P-kpoints).^2, 2));
  elseif strcmp(dist,'m') %distancia manhattan
    D = sum(abs(P-kpoints), 2);
  end
end

