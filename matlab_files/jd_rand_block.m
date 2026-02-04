function [varargout] = jd_rand_block(A, varargin)
%JD_RAND_BLOCK Block Jacobi-Davidson with sketched orthogonalization
%
%  Based on Balabanov-Grigori RGS framework. V is kept Theta-orthonormal,
%  meaning SV = Theta*V has l2-orthonormal columns.
%
%  [X, Lambda] = JD_RAND_BLOCK(A) computes the k smallest eigenvalues.
%
%  [X, Lambda] = JD_RAND_BLOCK(A, k) computes k eigenvalues.
%
%  [X, Lambda] = JD_RAND_BLOCK(A, k, opts) with options structure:
%    opts.tol          - Convergence tolerance (default: 1e-8)
%    opts.maxit        - Maximum iterations (default: 200)
%    opts.jmin         - Min subspace dimension (default: k+5)
%    opts.jmax         - Max subspace dimension (default: jmin+5)
%    opts.blk          - Block size (default: min(k, 4))
%    opts.v0           - Initial vectors (n x m matrix)
%    opts.sigma        - 'SR' (default), 'LR', 'SM', 'LM'
%    opts.M            - Preconditioner (default: [])
%    opts.disp         - Display level: 0 or 1 (default: 0)
%
%  Sketching options:
%    opts.sketch_type  - 'gaussian', 'rademacher', 'sparse' (default)
%    opts.sketch_size  - Sketch dimension s (default: 4*jmax)
%    opts.orth_method  - 'rcgs', 'rcgs2', 'rgs' (default)
%
%  [X, Lambda, history] = JD_RAND_BLOCK(...) also returns convergence history

% Parse inputs
[n, k, opts] = parse_inputs(A, varargin{:});
if n < 1, varargout = {[], [], []}; return; end

% Create sketching matrix
s = opts.sketch_size;
Theta = create_sketch_matrix(opts.sketch_type, s, n);

% Initialize parameters
tol = opts.tol / sqrt(k);
blk = opts.blk;
nmv = 0;
history = [];
nconv = 0;
nit = 0;
sigma = upper(opts.sigma);

% Converged eigenvectors (Theta-orthonormal)
Qschur = zeros(n, 0);
SQschur = zeros(s, 0);
Rschur = [];

% Initialize search space
[V, SV] = init_space(n, opts.v0, blk, Theta, Qschur, SQschur, opts);
j = size(V, 2);
W = A * V;
nmv = nmv + j;
SW = Theta * W;

% Sketched Rayleigh-Ritz matrix
M = SV' * SW;

% Main block JD loop
while nconv < k && nit < opts.maxit
    j = size(V, 2);
    
    % Solve projected eigenvalue problem
    [U, D] = eig(M);
    lambda = diag(D);
    
    % Sort by target
    lambda_real = real(lambda);
    switch sigma
        case 'LM', [~, I] = sort(-abs(lambda_real));
        case 'SM', [~, I] = sort(abs(lambda_real));
        case 'LR', [~, I] = sort(-lambda_real);
        otherwise, [~, I] = sort(lambda_real);
    end
    U = U(:, I);
    lambda = lambda(I);
    
    % Active block size
    p = min([blk, j, k - nconv]);
    
    % Block Ritz vectors
    Y = U(:, 1:p);
    X_blk = V * Y;
    SX_blk = SV * Y;
    W_blk = W * Y;
    Theta_vals = lambda(1:p);
    
    % Block residuals: R = A*X - X*diag(theta)
    R = W_blk - X_blk * diag(Theta_vals);
    SR = Theta * R;
    
    % Theta-orthogonalize residuals against converged space
    if ~isempty(Qschur)
        coeffs = SQschur \ SR;
        R = R - Qschur * coeffs;
        SR = Theta * R;
    end
    
    % Residual norms (Theta-norm)
    nr = vecnorm(SR);
    
    % Record history
    history = [history; min(nr), nit, nmv];
    if opts.disp
        fprintf('it=%d, nmv=%d, dim=%d, blk=%d, |r|=[%.2e, %.2e]\n', ...
            nit, nmv, j, p, min(nr), max(nr));
    end
    
    % Check convergence
    conv_idx = find(nr < tol);
    if ~isempty(conv_idx)
        [QY, ~] = qr(Y(:, conv_idx), 0);
        VQY = V * QY;
        
        Qschur = [Qschur, VQY];
        Rschur = [Rschur; Theta_vals(conv_idx)];
        SQschur = [SQschur, Theta * VQY];
        nconv = nconv + length(conv_idx);
        
        if nconv >= k, break; end
        
        % Deflate: keep unconverged Ritz vectors
        keep_idx = setdiff(1:j, conv_idx);
        if isempty(keep_idx)
            [V, SV] = init_space(n, [], blk, Theta, Qschur, SQschur, opts);
            j = size(V, 2);
            W = A * V;
            nmv = nmv + j;
            SW = Theta * W;
            M = SV' * SW;
        else
            U_keep = U(:, keep_idx);
            for i = 1:2
                U_keep = U_keep - QY * (QY' * U_keep);
            end
            [Q, ~] = qr(U_keep, 0);
            V = V * Q;
            W = W * Q;
            SV = SV * Q;
            SW = SW * Q;
            j = length(keep_idx);
            M = Q' * M * Q;
        end
        continue;
    end
    
    % Restart if subspace too large
    if j >= opts.jmax
        U_keep = U(:, 1:opts.jmin);
        [Q, ~] = qr(U_keep, 0);
        V = V * Q;
        W = W * Q;
        SV = SV * Q;
        SW = SW * Q;
        j = opts.jmin;
        M = Q' * M * Q;
    end
    
    % Block correction equation: project out converged + active Ritz vectors
    [SX_proj, RY] = qr(SX_blk, 0);
    X_proj = X_blk / RY;
    Q_proj = [Qschur, X_proj];
    SQ_proj = [SQschur, SX_proj];
    
    T = solve_correction(Q_proj, SQ_proj, Theta, R, opts);
    nit = nit + 1;
    
    % Expand subspace with correction block
    if ~isempty(Qschur)
        T = theta_orth_against(T, Qschur, SQschur, Theta, opts.orth_method);
    end
    T = theta_orth_against(T, V, SV, Theta, opts.orth_method);
    ST = Theta * T;
    
    % Whiten correction vectors
    [ST, RST] = qr(ST, 0);
    T = T / RST;
    
    if ~isempty(T)
        W_new = A * T;
        nmv = nmv + size(T, 2);
        SW_new = Theta * W_new;
        
        M = [M, SV' * SW_new; ST' * SW, ST' * SW_new];
        V = [V, T];
        SV = [SV, ST];
        W = [W, W_new];
        SW = [SW, SW_new];
    end
end

% Output
if nargout == 0
    disp(Rschur);
elseif nargout == 1
    varargout{1} = Rschur;
elseif nargout == 2
    varargout = {Qschur, diag(Rschur)};
else
    varargout = {Qschur, diag(Rschur), history};
end
end

%% =========================================================================
%% INPUT PARSING
%% =========================================================================

function [n, k, opts] = parse_inputs(A, varargin)
    n = size(A, 1);
    k = min(5, n);
    
    % Defaults
    opts.tol = 1e-8;
    opts.maxit = 200;
    opts.jmin = -1;
    opts.jmax = -1;
    opts.blk = -1;
    opts.v0 = [];
    opts.sigma = 'SR';
    opts.M = [];
    opts.disp = 0;
    opts.sketch_type = 'sparse';
    opts.sketch_size = -1;
    opts.orth_method = 'rgs';
    
    % Parse arguments
    for i = 1:length(varargin)
        arg = varargin{i};
        if isstruct(arg)
            fn = fieldnames(arg);
            for jj = 1:length(fn)
                opts.(fn{jj}) = arg.(fn{jj});
            end
        elseif isscalar(arg) && arg == round(arg) && arg > 0
            k = min(arg, n);
        end
    end
    
    % Set dependent defaults
    if opts.blk < 0, opts.blk = min(k, 4); end
    if opts.jmin < 0, opts.jmin = min(n, k + 5); end
    if opts.jmax < 0, opts.jmax = min(n, opts.jmin + 5); end
    if opts.sketch_size < 0, opts.sketch_size = max(4 * opts.jmax, 4 * k); end
end

%% =========================================================================
%% INITIALIZATION
%% =========================================================================

function [V, SV] = init_space(n, v0, blk, Theta, Qschur, SQschur, opts)
    if isempty(v0)
        V = randn(n, blk);
    else
        V = v0;
        if size(V, 2) < blk
            V = [V, randn(n, blk - size(V, 2))];
        end
    end
    
    % Theta-orthogonalize against converged space
    if ~isempty(Qschur)
        V = theta_orth_against(V, Qschur, SQschur, Theta, opts.orth_method);
    end
    
    % Theta-orthonormalize columns
    [V, SV] = sketch_orth(V, Theta, opts.orth_method);
end

%% =========================================================================
%% SKETCHING
%% =========================================================================

function Theta = create_sketch_matrix(sketch_type, s, n)
    switch lower(sketch_type)
        case 'gaussian'
            Theta = randn(s, n) / sqrt(s);
        case 'rademacher'
            Theta = (2 * (rand(s, n) > 0.5) - 1) / sqrt(s);
        case 'sparse'
            if exist('sparsestack', 'file') == 3
                Theta = sparsestack(s, n, 4);
            elseif exist('sparsesign', 'file') == 3
                Theta = sparsesign(s, n, 8);
            else
                Theta = sparsesign_backup(s, n, 8);
            end
        otherwise
            Theta = randn(s, n) / sqrt(s);
    end
end

%% =========================================================================
%% THETA-ORTHOGONALIZATION
%% =========================================================================

function [V, SV] = sketch_orth(A, Theta, method)
%SKETCH_ORTH Theta-orthonormalize columns of A
    [m, n] = size(A);
    V = zeros(m, n);
    SV = zeros(size(Theta, 1), n);
    
    % First column: just normalize
    V(:, 1) = A(:, 1);
    SV(:, 1) = Theta * V(:, 1);
    nsv = norm(SV(:, 1));
    V(:, 1) = V(:, 1) / nsv;
    SV(:, 1) = SV(:, 1) / nsv;
    
    % Remaining columns: orthogonalize then normalize
    for i = 2:n
        V(:, i) = A(:, i);
        
        switch lower(method)
            case 'rcgs'
                SV(:, i) = Theta * V(:, i);
                H = SV(:, 1:i-1)' * SV(:, i);
                V(:, i) = V(:, i) - V(:, 1:i-1) * H;
                
            case 'rcgs2'
                for pass = 1:2
                    SV(:, i) = Theta * V(:, i);
                    H = SV(:, 1:i-1)' * SV(:, i);
                    V(:, i) = V(:, i) - V(:, 1:i-1) * H;
                end
                
            otherwise  % 'rgs'
                SV(:, i) = Theta * V(:, i);
                H = SV(:, 1:i-1) \ SV(:, i);
                V(:, i) = V(:, i) - V(:, 1:i-1) * H;
        end
        
        % Normalize
        SV(:, i) = Theta * V(:, i);
        nsv = norm(SV(:, i));
        V(:, i) = V(:, i) / nsv;
        SV(:, i) = SV(:, i) / nsv;
    end
end

function V = theta_orth_against(V, Q, SQ, Theta, method)
%THETA_ORTH_AGAINST Theta-orthogonalize V against Q
    if isempty(Q) || isempty(V), return; end
    
    switch lower(method)
        case 'rcgs'
            SV = Theta * V;
            H = SQ' * SV;
            V = V - Q * H;
            
        case 'rcgs2'
            for pass = 1:2
                SV = Theta * V;
                H = SQ' * SV;
                V = V - Q * H;
            end
            
        otherwise  % 'rgs'
            SV = Theta * V;
            H = SQ \ SV;
            V = V - Q * H;
    end
end

%% =========================================================================
%% CORRECTION EQUATION
%% =========================================================================

function T = solve_correction(Q, SQ, Theta, R, opts)
%SOLVE_CORRECTION Block Olsen correction: T = (I - Q*Q'_Theta) * M^{-1} * R
    T = R;
    if ~isempty(opts.M)
        T = opts.M \ T;
    end
    T = theta_orth_against(T, Q, SQ, Theta, opts.orth_method);
end