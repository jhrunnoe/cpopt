classdef Placer < handle
  properties
    design;
    chip;
    nets;
    grid;
    masks;
    scale;

    % Electrostatic quantities
    density;
    potential;
    field;
    
    ix;       % indices of x coordinates (  1,   2, ...,   n)
    iy;       % indices of y coordinates (n+1, n+2, ..., 2*n)
    half_dx;
    half_dy;
    half_dz;

    uz;
    lz;

    mu    = 0.01; % Multiplier for energy 
    sigma = 1;    % Multiplier for bound constraint violation
  end
  properties(Constant)
    grid_resolution = 128;
    pad_factor = 0.50;
    step_contraction = 0.67;
    reduction_tolerance = 1.0e-04;
    df_tolerance = 1.0e-01;
    colors = colororder;
  end
  
  methods
    function [obj] = Placer()
      obj.design = generate_design();
      
      obj.nets = cell(obj.design.n_nets, 1);
      for j = 1:obj.design.n_nets
        obj.nets{j} = Net(j, obj.design.netlist{j});
      end
      
      obj.grid.nx = ceil(obj.grid_resolution*obj.design.R.dx/min(obj.design.R.dx, obj.design.R.dy));
      obj.grid.ny = ceil(obj.grid_resolution*obj.design.R.dy/min(obj.design.R.dx, obj.design.R.dy));
      obj.grid.pad.x = ceil(obj.pad_factor*obj.grid.nx);
      obj.grid.pad.y = ceil(obj.pad_factor*obj.grid.ny);
      obj.grid.pad.z = [obj.grid.pad.x*ones(obj.design.n_cells, 1); obj.grid.pad.y*ones(obj.design.n_cells,1)];
      obj.grid.total.x = obj.grid.nx + 2*obj.grid.pad.x;
      obj.grid.total.y = obj.grid.ny + 2*obj.grid.pad.y;

      obj.scale.x = obj.grid.nx/obj.design.R.dx;
      obj.scale.y = obj.grid.ny/obj.design.R.dy;

      obj.chip.R  = struct('x', 0, 'y', 0, 'dx', obj.grid.nx, 'dy', obj.grid.ny);
      obj.chip.x0 = obj.scale.x*obj.design.x0;
      obj.chip.y0 = obj.scale.y*obj.design.y0;
      obj.chip.z0 = [obj.chip.x0; obj.chip.y0];
      obj.chip.dx = obj.scale.x*obj.design.dx;
      obj.chip.dy = obj.scale.y*obj.design.dy;
      obj.chip.dz = [obj.chip.dx; obj.chip.dy];
       
      obj.ix = 1:obj.design.n_cells;
      obj.iy = (obj.design.n_cells + 1):2*obj.design.n_cells;
      obj.half_dx = 0.50*obj.chip.dx;
      obj.half_dy = 0.50*obj.chip.dy;
      obj.half_dz = [obj.half_dx; obj.half_dy];

      obj.lz = obj.half_dz;
      obj.uz = [obj.chip.R.dx*ones(obj.design.n_cells, 1); 
                obj.chip.R.dy*ones(obj.design.n_cells, 1)] - obj.half_dz;

      [obj.density, obj.masks] = obj.compute_density(obj.chip.x0, obj.chip.y0);

      obj.potential = zeros(obj.grid.nx, obj.grid.ny);
      obj.field.x   = zeros(obj.grid.nx, obj.grid.ny);
      obj.field.y   = zeros(obj.grid.nx, obj.grid.ny);
    end

    function [] = plot_cells(obj, x, y, cells)
      if isempty(cells)
        cells = 1:obj.design.n_cells;
      end
      rectangle('Position', [obj.chip.R.x,  obj.chip.R.y,   ...
                             obj.chip.R.dx, obj.chip.R.dy], ...
                             'EdgeColor', obj.colors(1, :), 'FaceColor', [obj.colors(1,:) 0.05], "LineStyle","--");
      hold on;
      for i = cells
        color = 'black';
        for j = 1:7
          if ismember(i, obj.nets{j}.cells)
            color = obj.colors(j, :);
            break;
          end  
        end
        rectangle('Position', [x(i) - obj.half_dx(i), ...
                               y(i) - obj.half_dy(i), ...
                               obj.chip.dx(i), ...
                               obj.chip.dy(i)], 'EdgeColor', color);
      end
      axis([-obj.grid.pad.x, obj.chip.R.dx + obj.grid.pad.x, -obj.grid.pad.y, obj.chip.R.dy + obj.grid.pad.y]);
      xline(0, "LineStyle",":");
      yline(0, "LineStyle",":");

      hold off;
    end

    function [hm] = plot_heatmap(obj, M)
      hm = imagesc(M');
      set(gca, 'Ydir', 'normal');
      hm.XData = hm.XData - obj.grid.pad.x - 1;
      hm.YData = hm.YData - obj.grid.pad.y - 1;
      colormap('parula');
      colorbar;
      rectangle('Position', [obj.chip.R.x,  obj.chip.R.y,   ...
                             obj.chip.R.dx, obj.chip.R.dy], ...
                             'EdgeColor', obj.colors(1, :), "LineStyle","--");
      
      axis([-obj.grid.pad.x, obj.chip.R.dx + obj.grid.pad.x - 1, -obj.grid.pad.y, obj.chip.R.dy + obj.grid.pad.y - 1]);
    end

    function [clearance] = compute_clearance(obj, x, y, cells)
      %% ----------------- compute_clearance()
      %  Computation of the minimum pairwise clearance for the provided
      %  group of cells. The result is positive if there is no overlap,
      %  zero when two or more cells are touching, and negative if two or
      %  more cells intersect.
      
      pairs   = nchoosek(cells, 2);
      n_pairs = size(pairs, 1);
      deltas  = zeros(n_pairs, 1);

      for k = 1:n_pairs
        xi = x(pairs(k, 1)); 
        xj = x(pairs(k, 2));
        yi = y(pairs(k, 1)); 
        yj = y(pairs(k, 2));

        dxi = obj.chip.dx(pairs(k, 1));
        dxj = obj.chip.dx(pairs(k, 2));
        dyi = obj.chip.dy(pairs(k, 1));
        dyj = obj.chip.dy(pairs(k, 2));
        
        dijx = max(xi - 0.50*dxi, xj - 0.50*dxj) - min(xi + 0.50*dxi, xj + 0.50*dxj);
        dijy = max(yi - 0.50*dyi, yj - 0.50*dyj) - min(yi + 0.50*dyi, yj + 0.50*dyj);

        deltas(k) = max(dijx, dijy);
      end
      clearance = min(deltas);
    end

    function [result] = solve(obj)
      z = obj.chip.z0;

      converged = false;
      [f, g, ~, ~] = obj.evaluate(z);
      
      while ~converged
        p = -g;
        [z1, f1, g1, D, M] = obj.search(z, p, f, g);

        df = f1-f
        
        if abs(df) < obj.df_tolerance
          converged = true;
        end

        % Set the new base point
        obj.density = D;
        obj.masks = M;
        z = z1;
        f = f1;
        g = g1;
        
      end
      result.x = z(obj.ix);
      result.y = z(obj.iy);
      result.z = z;
      result.f = f;
    end

    function [z1, f1, g1, D, M] = search(obj, z, p, f, g)
      
      reduced = false;
      alpha = 1;
      gTp = g'*p;

      while ~reduced && alpha > eps
        z1 = z + alpha*p;

        z1 = median([obj.lz - obj.grid.pad.z + 1, z1, obj.uz + obj.grid.pad.z - 1], 2);

        [f1, g1, D, M] = obj.evaluate(z1);
        
        if f - f1 >= -obj.reduction_tolerance*alpha*gTp
          reduced = true;
        else
          alpha = obj.step_contraction*alpha;
        end
      end
    end

    function [f, g, D, M] = evaluate(obj, z)
      %% ------------ evaluate()
      %
      %  DESCRIPTION
      %  This computes the objective function 
      %       f(z) = W(z) + mu*E(z)
      %  and its gradient, where W is some measure of the
      %  wire length and E is the system electrostatic energy.
      %  Possible choices for W(z) include the weighted average
      %  smooth approximation of half-perimeter wire length, or
      %  the pin coordinate variances.
      %  
      %  Parameters
      %  z: the current placement coordinates z = [x; y].
      %  D: the previous density matrix  - will be updated to reflect the new placement z.
      %  M: the previous occupancy masks - will be updated to reflect the new placement z.
      % 
      %  Returns
      %  f: the function value.
      %  g: the gradient of f at z.
      %  D: updated density matrix.
      %  M: updated occupancy masks.
      
      x = z(obj.ix); 
      y = z(obj.iy);

      [D, M]    = obj.compute_density(x, y);

      P   = sparse(obj.grid.total.x, obj.grid.total.y);
      F.x = sparse(obj.grid.total.x, obj.grid.total.y);
      F.y = sparse(obj.grid.total.x, obj.grid.total.y);
      
      % If any cells extend out of the main (grid.nx by grid.ny)
      % grid, extend the grid into the pad to capture all cells.
      xmin = min([1; floor(x - obj.half_dx) + 1]);
      xmax = max([obj.grid.nx; ceil(x + obj.half_dx)]);
      ymin = min([1; floor(y - obj.half_dy) + 1]);
      ymax = max([obj.grid.ny; ceil(y + obj.half_dy)]);
      I = obj.grid.pad.x + (xmin:xmax);
      J = obj.grid.pad.y + (ymin:ymax);
      
      [~, F_, E] = obj.compute_electrostatics(D(I, J));

      F.x(I, J) = F_.x;
      F.y(I, J) = F_.y;

      % Determine in which grid elements each cell locates, extract 
      % the electric field at those locations, and scale by the cell 
      % charges (area):
      dE.x = zeros(size(x));
      dE.y = zeros(size(y));

      for k = 1:obj.design.n_cells
        [I, J] = find(M{k});
        % The total force on cell k will be the sum of the forces
        % due to the electrostatic field at each grid element scaled by the overlap
        % area there.
        dE.x(k) = -sum(F.x(sub2ind(size(F.x), I, J)), "all");
        dE.y(k) = -sum(F.y(sub2ind(size(F.y), I, J)), "all");
      end

      % The wirelength gradient is the sum over nets 
      dW.x = zeros(size(x));
      dW.y = zeros(size(y));
      W = 0;
      for j = 1:obj.design.n_nets
        [nW, ndW] = obj.nets{j}.pin_spread(x, y);
        % [nW, ndW] = obj.nets{j}.WAWL(x, y);
        W = W + nW;
        dW.x(obj.nets{j}.cells) = dW.x(obj.nets{j}.cells) + ndW.x;
        dW.y(obj.nets{j}.cells) = dW.y(obj.nets{j}.cells) + ndW.y;
      end
      
      % Add the norm of the bound constraint residuals to the
      % objective and its gradient
      
      upr_resid = max(z - obj.uz, 0);
      low_resid = max(obj.lz - z, 0);

      R = 0.50*norm( upr_resid )^2 + 0.50*norm( low_resid )^2;
      dR = upr_resid - low_resid;

      f = W + obj.mu*E + obj.sigma*R;
      g = [dW.x + obj.mu*dE.x;
           dW.y + obj.mu*dE.y] + obj.sigma*dR;
    end

    function [D, M] = update_density(obj, x, y, cells)
      %% ------------ update_density()
      %
      %  Description
      %  Updates the density matrix D and occupancy masks M based
      %  on new coordinates x, y. It is assumed only cells listed
      %  change position. The update is done by subtracting out
      %  the density contribution of the moved cells, updating
      %  their sparse density mask, and adding the new
      %  contribution back into D. 
      %
      %  Parameters
      %  x    : new cell coordinates
      %  y    : new cell coordinates
      %  D    : (sparse double) current density matrix.
      %  M    : (cell array) cell array of sparse occupancy masks.
      %  cells: list of cells to update.
      %
      %  Returns
      %  D: (sparse double) Density matrix
      %  M: (cell array) Grid-cell occupancy masks
      %
      D = obj.density;
      M = obj.masks;
      for k = cells
        [I, J, v] = find(obj.masks{k});
        di = floor(x(k) - obj.half_dx(k)) - I(1) + 1;
        dj = floor(y(k) - obj.half_dy(k)) - J(1) + 1; 
        if di ~= 0 || dj ~= 0
          D    = D - obj.masks{k};
          M{k} = sparse(I + di, J + dj, v, obj.grid.nx, obj.grid.ny);
          D    = D + M{k};
        end
      end
    end

    function [D, M] = compute_density(obj, x, y)
      %% ------------ compute_density()
      %
      %  Description
      %  Compute the density based on the placement coordinates
      %  x, y. The instances listed in the cells parameter are
      %  assumed to have moved relative to the coordinates used
      %  to compute the provided D, M. If cells lists all design
      %  cells, the density is computed from scratch, otherwise
      %  the density is updated.
      %
      %  Parameters
      %  x    : new cell coordinates
      %  y    : new cell coordinates
      %  D    : (sparse double) current density matrix.
      %  M    : (cell array) cell array of sparse occupancy masks.
      %  cells: list of cells to update.
      %
      %  Returns
      %  D: (sparse double) Density matrix
      %  M: (cell array) Grid-cell occupancy masks
      %

      D = sparse(obj.grid.total.x, obj.grid.total.y);
      M = cell(obj.design.n_cells, 1);
      for k = 1:obj.design.n_cells
        li = floor(x(k) - obj.half_dx(k)) + 1;
        ui = ceil(x(k)  + obj.half_dx(k));
        lj = floor(y(k) - obj.half_dy(k)) + 1;
        uj = ceil(y(k)  + obj.half_dy(k));
        
        M{k} = sparse(obj.grid.total.x, obj.grid.total.y);
        M{k}(obj.grid.pad.x + (li:ui), obj.grid.pad.y + (lj:uj)) = 1;
        % The density is the sum over individual grid-cell occupancy masks
        D = D + M{k};
      end
    end

    function [P, F, E] = compute_electrostatics(obj, D)
      %% --------------- compute_electrostatics()
      %
      %  Description
      %  Computes the electrostatic potential map, horizontal and
      %  vertical field maps, and total energy using fast Fourier
      %  transforms
      %
      %  Parameters
      %  D: density matrix
      %
      %  Returns
      %  P: (double) potential map such that P, D satisfy Poisson's equation
      %  F: (struct) electric field maps F.x and F.y
      %  E: (scalar) total system energy

      if issparse(D)
        D = full(D);
      end
      [M, N] = size(D);
      % Generate the mesh of support for the cosine basis
      u  = (pi/M)*(0:(M - 1));
      v  = (pi/N)*(0:(N - 1));

      % Compute the Fourier coefficients to express the density D in the
      % cosine basis. The 2D DCT is achieved by iterating 1D DCT
      A = dct(dct(D, [], dim=1), [], dim=2);

      % Compute the electrostatic potential: (the coefficients
      % A./S come from rho(x, y) = -laplacian(psi(x, y)), writing the expansion
      % of the density in the cosine basis, and integrating twice
      % in each variable.
      S       = (u.^2)' + v.^2; % (S)um of squares
      S(1, 1) = 1;
      C       = A./S;
      C(1, 1) = 0;

      P = idct(idct(C, [], dim=1), [], dim=2);

      % The force calculation can also be done using the idst function:
      % T   = idct(Cu')';
      % F.x = idst([T(2:end, :); zeros(1, obj.grid.ny)]);
      % F.y = idct(idst([Cv(:, 2:end) zeros(obj.grid.nx, 1)]')');

      % Differentiating the potential map with respect to x gives the 
      % horizontal component of the electrostatic field. This multiplies
      % each row of the coefficient matrix C by elements of u.
      Cu  = u'.*C;
      F.x = obj.idxst(idct(Cu, [], "dim", 2));

      % Differentiating the potential map with respect to y gives the
      % vertical component of the electrostatic field.
      Cv  = C.*v;
      F.y = idct(obj.idxst(Cv, [], "dim", 2));

      % Total system energy is half the sum over all charge-potential
      % products (charge represented by area)
      E = 0.50*sum(P.*D, "all");
    end
  end

  methods(Static)
    function [area] = compute_overlap(r1, r2)
      %% ------------ compute_overlap(r1, r2)
      % Compute the area of the overlap of two rectangles r1, r2. 
      dx = max(0, min(r1.x + r1.dx, r2.x + r2.dx) - max(r1.x, r2.x));
      dy = max(0, min(r1.y + r1.dy, r2.y + r2.dy) - max(r1.y, r2.y));
      area = dx*dy;
    end

    function [Y] = idxst(X, ~, varargin)
      %% --------- idxst(X, ~, varargin)
      % Purpose: 
      % Compute the inverse discrete sine transform of the input
      % transformed by [a_1, a_2, a_3,..., a_n] -> [a_2, a_3,..., a_n, 0]
      % in terms of the idct function (by reversing the input and flipping 
      % the sign of every other output).
      if nargin > 2
        dim = varargin{2};
      else
        dim = 1;
      end
      if dim == 1
        X = X';
      end
      [m, ~] = size(X);
      Y = zeros(size(X));
      for i = 1:m
        y = idct(flip([X(i, 2:end) 0]));
        y(2:2:end) = -y(2:2:end);
        Y(i, :) = y;
      end
      if dim == 1
        Y = Y';
      end
    end
  end
end