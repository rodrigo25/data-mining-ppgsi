function [ U, kpoints ] = kmeans( data, k, dist )

  [n, m] = size(data);

  % INICIA PROTÓTIPOS RANDOMICAMENTE
  maximo = max(data);   %valores máximos de cada variavel
  minimo = min(data);   %valores minimos de cada variavel
  kpoints = zeros(k,m);
  for i=1:m     % cria k valores para a variavel m no intervalo [min max] de m
    kpoints(:,i) = minimo(i)+(maximo(i)-minimo(i)).*rand(k,1);
  end

  % DEFINIÇÃO DE VARIÁVEIS
  maxIt = 100; % maximo de iterações
  it = 0; %contador de iterações
  U = zeros(k, n); %cria matriz de partição
  old_kpoints = kpoints+5; %protótipos anteriores para controle
  epsilon = 0; %erro máximo
  
  % LAÇO DE ITERAÇÕES DO K-MEANS
  while(abs(old_kpoints - kpoints)>epsilon & (it<maxIt))
    it = it+1;
    old_kpoints = kpoints;
    
    % ATUALIZA MATRIZ DE PARTIÇÃO
    for i=1:n   %para cada instancia
      p = data(i,:);
      D = distance(p, kpoints, k, dist); %calcula distancias dos protótipos
      [~,ind] = min(D); %Verifica qual o protótipo de menor distancia
      U(:,i)=0; %Zera todos os protótipos desse ponto
      U(ind,i)=1; %atribui 1 ao protótipo de menor distancia
    end
          
    % PLOTA GRÁFICO (Para problemas bidimensionais)        
    %plotClusters2D(data,kpoints,U,k,[]);
    %pause;
    
    % ATUALIZA PROTÓTIPOS
    for i=1:k  %para cada protótipo
      kpoints(i,:) = (U(i,:)*data)/length(find(U(i,:)==1)); %atualiza com a média
    end

  end
  
  
  % ÚLTIMA ATUALIZAÇÃO DA MATRIZ
  for i=1:n   %para cada instancia
    p = data(i,:);
    D = distance(p, kpoints, k, dist); %calcula distancias dos protótipos
    [~,ind] = min(D); %Verifica qual o protótipo de menor distancia
    U(:,i)=0; %Zera todos os protótipos desse ponto
    U(ind,i)=1; %atribui 1 ao protótipo de menor distancia
  end
  
  fprintf('número de iterações: %i\n\n', it)
  
  
end


% FUNÇÃO DE DISTÂNCIAS
function D = distance(p, kpoints, k, dist)
  P = repmat(p,k,1);
  if strcmp(dist,'e') %distancia euclidiana
    D = sqrt(sum((P-kpoints).^2, 2));
  elseif strcmp(dist,'m') %distancia manhattan
    D = sum(abs(P-kpoints), 2);
  end
end

