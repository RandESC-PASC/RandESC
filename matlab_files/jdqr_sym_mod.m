function [varargout] = jdqr_sym_mod(A, varargin)
%JDQR_SYM Jacobi-Davidson for symmetric matrices (smallest/largest eigenvalues)
%
%  [X, Lambda] = JDQR_SYM(A) computes the k smallest eigenvalues
%  and corresponding eigenvectors of symmetric matrix A.
%
%  [X, Lambda] = JDQR_SYM(A, k) computes k eigenvalues.
%
%  [X, Lambda] = JDQR_SYM(A, k, opts) with options structure:
%    opts.tol      - Convergence tolerance (default: 1e-8)
%    opts.maxit    - Maximum iterations (default: 200)
%    opts.jmin     - Min subspace dimension (default: k+5)
%    opts.jmax     - Max subspace dimension (default: jmin+5)
%    opts.v0       - Initial vectors (n x m matrix)
%    opts.sigma    - 'SR' (default), 'LR', 'SM', 'LM'
%    opts.M        - Preconditioner matrix (default: [])
%    opts.disp     - Display progress: 0,1 (default: 0)
%
%  [X, Lambda, history] = JDQR_SYM(...) also returns convergence history

global nmv Qschur Rschur

% Parse inputs
[n, k, opts] = parse_inputs(A, varargin{:});
if n < 1, varargout = {[],[],[]}; return; end

% Initialize
tol = opts.tol / sqrt(k);
Qschur = zeros(n, 0);
Rschur = [];
nmv = 0;
history = [];
nconv = 0;
nit = 0;
sigma = upper(opts.sigma);

% Initialize search space
V = init_space(n, opts.v0);
j = size(V, 2);
W = matvec(A, V);
M = V' * W;

% Main JD loop
nlit = 0;
while nconv < k && nit < opts.maxit
    
    % Compute Ritz pairs (symmetric: M = U * D * U')
    [U, D] = eig(M);
    lambda = diag(D);
    
    % Sort eigenvalues
    switch sigma
        case 'LM', [~, I] = sort(-abs(lambda));
        case 'SM', [~, I] = sort(abs(lambda));
        case 'LR', [~, I] = sort(-lambda);
        otherwise, [~, I] = sort(lambda);  % SR
    end
    
    % Simple permutation (valid for symmetric/diagonal D)
    U = U(:, I);
    D = D(I, I);
    
    theta = D(1, 1);
    u = V * U(:,1);
    w = W * U(:,1);
    r = w - theta * u;
    r = orth_against(r, Qschur);
    nr = norm(r);
    
    % Record history
    history = [history; nr, nit, nmv];
    if opts.disp
        fprintf('it=%d, nmv=%d, dim=%d, |r|=%.2e, theta=%.6e\n', nit, nmv, j, nr, theta);
    end
    
    % Check convergence
    if nr < tol
        s = Qschur' * w;
        Qschur = [Qschur, u];
        Rschur = [Rschur, s; zeros(1, nconv), theta];
        nconv = nconv + 1;
        
        if opts.disp
            fprintf('  >>> Eigenvalue %d converged: %.10e\n', nconv, theta);
        end
        
        if nconv >= k, break; end
        
        % Deflate
        if j == 1
            V = init_space(A, n, [], opts.jmin);
            j = size(V, 2);
            W = matvec(A, V);
            M = V' * W;
        else
            J = 2:j; j = j - 1;
            U = U(:, J); D = D(J, J);
            V = V * U; W = W * U;
            M = D;
        end
        nlit = 0;
        continue;
    end
    
    % Restart if needed
    if j >= opts.jmax
        J = 1:opts.jmin;
        U = U(:, J); D = D(J, J);
        V = V * U; W = W * U;
        M = D; j = opts.jmin;
    end
    
    % Solve correction equation
    Q_proj = [Qschur, u];
    t = solve_correction(Q_proj, r, opts);
    nlit = nlit + 1;
    nit = nit + 1;
    
    % Expand subspace
    t = orth_against(t, Qschur);
    t = orth_against(t, V);
    nt = norm(t);
    if nt > 1e-14
        t = t / nt;
        w = matvec(A, t);
        M = [M, V'*w; t'*W, t'*w];
        V = [V, t]; W = [W, w];
        j = j + 1;
    end
end

% Output
if nargout == 0
    disp(diag(Rschur));
elseif nargout == 1
    varargout{1} = diag(Rschur);
elseif nargout == 2
    varargout = {Qschur, diag(diag(Rschur))};
else
    varargout = {Qschur, diag(diag(Rschur)), history};
end
end

%% === Helper Functions ===

function [n, k, opts] = parse_inputs(A, varargin)
n = size(A, 1);
k = min(5, n);

opts.tol = 1e-8;
opts.maxit = 200;
opts.jmin = -1;
opts.jmax = -1;
opts.v0 = [];
opts.sigma = 'SR';
opts.M = [];
opts.disp = 0;

for i = 1:length(varargin)
    arg = varargin{i};
    if isstruct(arg)
        fn = fieldnames(arg);
        for jj = 1:length(fn), opts.(fn{jj}) = arg.(fn{jj}); end
    elseif isscalar(arg) && arg == round(arg) && arg > 0
        k = min(arg, n);
    end
end

if opts.jmin < 0, opts.jmin = min(n, k + 5); end
if opts.jmax < 0, opts.jmax = min(n, opts.jmin + 5); end
end

function V = init_space(n, v0)
    global Qschur
    
    if isempty(v0)
        V = ones(n, 1) + 0.1 * rand(n, 1);
    else
        V = v0;
    end
    
    V = orth_against(V, Qschur);
    [V, ~] = qr(V, 0);
end

function v = orth_against(v, Q)
    if isempty(Q) || size(v,1) == 0, return; end
    for rep = 1:2
        v = v - Q * (Q' * v);
    end
end

function w = matvec(A, v)
    global nmv
    w = A * v;
    nmv = nmv + size(v, 2);
end

function t = solve_correction(Q, r, opts)
    t = r;
    if ~isempty(opts.M), t = opts.M \ t; end
    t = t - Q * (Q' * t);
end

