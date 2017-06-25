function [Ynew, classes] = multiclassY(Y)
%multiclassY transforma os rotulos da matriz Y (n x 1) em uma matriz Ynew
% (n x nc) onde nc = numero de classes em Y. Cada elemento em Ynew indica
% se a n-esima linha pertence a classe nc (1) ou se nao pertence (0)
%
%	Y - a matriz (n x 1) de entrada, contendo nc classes
    classes = unique(Y);
    nc = length(classes);
    Ynew = zeros(size(Y,1), nc);
    for cInd=1:nc
        c = classes(cInd);
        Ynew(Y == c, cInd) = 1;
    end
end