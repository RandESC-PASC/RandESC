function [varargout] = jdqr_sketched_v1(A, varargin)
%JDQR_SKETCHED_V1 Jacobi-Davidson with sketched orthogonalization
%
%  Based on Balabanov-Grigori RGS framework. V is kept Theta-orthonormal,
%  meaning SV = Theta*V has l2-orthonormal columns.
%
%  [X, Lambda] = JDQR_SKETCHED_V1(A, k, opts)
%
%  Options:
%    opts.tol          - Convergence tolerance (default: 1e-8)
%    opts.maxit        - Maximum iterations (default: 200)
%    opts.jmin         - Min subspace dimension (default: k+5)
%    opts.jmax         - Max subspace dimension (default: jmin+5)
%    opts.v0           - Initial vector
%    opts.sigma        - 'SR' (smallest real, default), 'LR', 'SM', 'LM'
%    opts.M            - Preconditioner (default: [])
%    opts.disp         - Display: 0,1 (default: 0)
%
%  Sketching options:
%    opts.sketch_type  - 'gaussian', 'rademacher', 'sparse' (default)
%    opts.sketch_size  - Sketch dimension s (default: 4*jmax)
%    opts.orth_method  - 'rcgs', 'rcgs2', 'rgs' (default)

global nmv Qschur Rschur SQschur

[n, k, opts] = parse_inputs(A, varargin{:});
if n < 1, varargout = {[],[],[]}; return; end

% Sketching matrix
s = opts.sketch_size;
Theta = create_sketch_matrix(opts.sketch_type, s, n);

% Initialize
tol = opts.tol / sqrt(k);
nmv = 0;
history = [];
nconv = 0;
nit = 0;
sigma = upper(opts.sigma);

% Converged eigenvectors (Theta-orthonormal)
Qschur = zeros(n, 0);
SQschur = zeros(s, 0);
Rschur = [];

% Initialize search space with Arnoldi expansion
[V, SV] = init_search_space(n, opts.v0, Theta, Qschur, SQschur, opts);
j = size(V, 2);
W = matvec(A, V); % can get rid of storing W entirely if Au is computed at each iteration (1 extra matvec)
SW = Theta * W;

% Sketched Rayleigh-Ritz matrix
M = SV' * SW;

nlit = 0;
while nconv < k && nit < opts.maxit
    j = size(V, 2);
    
    % Solve eigenvalue problem
    [U, D] = eig(M);
    lambda = diag(D);
    lambda_real = real(lambda);
    
    % Sort by target
    switch sigma
        case 'LM', [~, I] = sort(-abs(lambda_real));
        case 'SM', [~, I] = sort(abs(lambda_real));
        case 'LR', [~, I] = sort(-lambda_real);
        otherwise, [~, I] = sort(lambda_real);
    end
    U = U(:, I);
    lambda = lambda(I);
    
    % Best Ritz pair
    y = U(:, 1);
    y = y / norm(y);
    
    u = V * y;
    su = SV * y;
    w = W * y;
    sw = SW * y;
    
    % Sketched Rayleigh quotient: theta = (Theta*u)' * (Theta*w) / ||Theta*u||^2
    theta = real(su' * sw) / (su' * su);
    % theta = real(u'*w)/(u'*u); % true rayleigh quotient doesnt work

    % Residual
    r = w - theta * u;
    sr = Theta * r;
    % Theta-orthogonalize residual against converged space
    if ~isempty(Qschur) % RGS
        coeffs = SQschur \ sr;
        r = r - Qschur * coeffs;
        sr = sr - SQschur * coeffs;
    end
    nr = norm(sr); % if this is not here, it does not converge

    % History
    history = [history; nr, nit, nmv];
    if opts.disp
        fprintf('it=%d, nmv=%d, dim=%d, |r|=%.2e, theta=%.6e\n', nit, nmv, j, nr, theta);
    end
    
    % Check convergence
    if nr < tol       
        s_col = Qschur' * w; 
        Qschur = [Qschur, u];
        SQschur = [SQschur, su];
        Rschur = [Rschur, s_col; zeros(1, nconv), theta];
        
        nconv = nconv + 1;
        
        if opts.disp >= 1
            fprintf('  >>> Eigenvalue %d converged: %.10e\n', nconv, theta);
        end
        
        if nconv >= k, break; end
        
        if j > 1
            
            % Project u1 out of remaining eigenvectors
            U_rest = U(:, 2:j);
            for i = 1:2 % CGS2
                U_rest = U_rest - y *(y'*U_rest);
            end

            % l2-orthonormalize the result
            [Q, ~] = qr(U_rest, 0);

            V = V * Q;
            W = W * Q;
            SV = SV * Q;
            SW = SW * Q;
            j = j-1;
            M = Q'*M*Q;
        else
            % Only 1 vector: cold restart
            [V, SV] = init_search_space(A, n, [], 1, Theta, Qschur, SQschur, opts);
            j = size(V, 2);
            W = matvec(A, V);
            SW = Theta * W;
            M = SV' * SW;
        end
        
        nlit = 0;
        continue;
    end
    
    % Restart if subspace too large
    if j >= opts.jmax
        U_keep = U(:, 1:opts.jmin);
        [Q, ~] = qr(U_keep, 0);  % l2-orthonormalize coefficients
        
        V = V * Q;
        W = W * Q;
        SV = SV * Q;  % Recompute sketches
        SW = SW * Q;
        j = opts.jmin;
        M = Q'*M*Q;
    end
    
    % Correction equation
    % Note: u needs to be Theta-normalized for projection
    nu = norm(su);
    u_proj = u / nu;
    su_proj = su / nu;
    
    Q_proj = [Qschur, u_proj];
    SQ_proj = [SQschur, su_proj];
    
    t = solve_correction(Q_proj, SQ_proj, Theta, r, opts);
    nlit = nlit + 1;
    nit = nit + 1;
    
    % Expand subspace
    % Theta-orthogonalize against converged space
    if ~isempty(Qschur)
        t = theta_orth_against(t, Qschur, SQschur, Theta, opts.orth_method);
    end
    % Theta-orthogonalize against current V
    t = theta_orth_against(t, V, SV, Theta, opts.orth_method);
    st = Theta * t;
    nt = norm(st);
    
    
    if nt > 1e-14
        t = t / nt;
        st = st / nt;
        
        wt = matvec(A, t);
        swt = Theta * wt;
        
        M = [M, SV' * swt; st' * SW, st' * swt];
        
        V = [V, t];
        SV = [SV, st];
        W = [W, wt];
        SW = [SW, swt];
    end
end

% Output
eigenvalues = diag(Rschur);
if nargout == 0
    disp(eigenvalues);
elseif nargout == 1
    varargout{1} = eigenvalues;
elseif nargout == 2
    varargout = {Qschur, diag(eigenvalues)};
else
    varargout = {Qschur, diag(eigenvalues), history};
end
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
            if exist('sparsestack','file') == 3
                Theta = sparsestack(s, n, 4);
            elseif exist('sparsesign','file') == 3
                Theta = sparsesign(s, n, 8);
            else
                Theta = sparsesign_backup(s, n, 8);
            end
        otherwise
            Theta = randn(s, n) / sqrt(s);
    end
end

function v = theta_orth_against(v, V, SV, Theta, method)
%THETA_ORTH_AGAINST Theta-orthogonalize v against V
%
%  Methods:
%    'rcgs'  - Randomized CGS (single pass)
%    'rcgs2' - Randomized CGS with reorthogonalization (two passes)
%    'rgs'   - Randomized GS via least squares

    if isempty(V), return; end
    
    switch lower(method)
        case 'rcgs'
            sv = Theta * v;
            h = SV' * sv;
            v = v - V * h;
            
        case 'rcgs2'
            for pass = 1:2
                sv = Theta * v;
                h = SV' * sv;
                v = v - V * h;
            end
            
        case 'rgs'
            sv = Theta * v;
            h = SV \ sv;
            v = v - V * h;
            
        otherwise  % default rgs
            sv = Theta * v;
            h = SV \ sv;
            v = v - V * h;
    end
end

%% =========================================================================
%% INITIALIZATION
%% =========================================================================

function [V, SV] = init_search_space(n, v0, Theta, Qschur, SQschur, opts)
%INIT_SEARCH_SPACE Initialize search space with a single vector
%
%  JD correction naturally expands the subspace - no need for Arnoldi.
%  This saves (jmin-1) matvecs at startup.

    if isempty(v0)
        v = randn(n, 1);
    else
        v = v0(:, 1);
    end
    
    % Theta-orthogonalize against converged space
    if ~isempty(Qschur)
        v = theta_orth_against(v, Qschur, SQschur, Theta, opts.orth_method);
    end
    
    % Normalize to unit Theta-norm
    sv = Theta * v;
    nv = norm(sv);
    if nv < 1e-14
        v = randn(n, 1);
        if ~isempty(Qschur)
            v = theta_orth_against(v, Qschur, SQschur, Theta, opts.orth_method);
        end
        sv = Theta * v;
        nv = norm(sv);
    end
    v = v / nv;
    sv = sv / nv;
    
    V = v;
    SV = sv;
end

%% =========================================================================
%% CORRECTION EQUATION
%% =========================================================================

function t = solve_correction(Q, SQ, Theta, r, opts)
%OLSEN_CORRECTION Olsen-style correction
    t = r;
    if ~isempty(opts.M)
        t = opts.M \ t;
    end
    % Project out Q using Theta inner product
    t = theta_orth_against(t, Q, SQ, Theta, opts.orth_method);
end

%% =========================================================================
%% UTILITIES
%% =========================================================================

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
    opts.sketch_type = 'sparse';
    opts.sketch_size = -1;
    opts.orth_method = 'rgs';  % Default: reorthogonalized
    
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
    
    if opts.jmin < 0, opts.jmin = min(n, k + 5); end
    if opts.jmax < 0, opts.jmax = min(n, opts.jmin + 5); end
    if opts.sketch_size < 0, opts.sketch_size = max(4 * opts.jmax, 4 * k); end
end

function w = matvec(A, v)
    global nmv
    w = A * v;
    nmv = nmv + size(v, 2);
end
