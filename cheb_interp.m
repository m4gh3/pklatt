% find 10 Chebyshev nodes and mark them on the plot
x = -1:1e-3:1;
n = 3;
k = 1:3; % iterator
xc = cos((2*k-1)/2/n*pi); % Chebyshev nodes
yc = exp(xc); % function evaluated at Chebyshev nodes
hold on;
plot(xc, yc, 'o' )

% find polynomial to interpolate data using the Chebyshev nodes
p = polyfit(xc, yc, n-1 ); % gives the coefficients of the polynomial of degree 2
plot(x, polyval(p,x), '--' ); % plot polynomial
plot(x, exp(x) ); %plot exp
p2 = [0.5, 1, 1]
plot(x, polyval(p2,x), '--' );
sqrt(p)
