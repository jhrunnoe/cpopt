% rng('default');

p = Placer();
[P, F, E] = p.compute_electrostatics(p.density);

p.plot_heatmap(P);
p.plot_cells(p.design.x0, p.design.y0, 1:p.design.n_cells);

figure; 
p.plot_heatmap(F.x);
 
figure; 
p.plot_heatmap(F.y);
 
figure; 
p.plot_heatmap(P); 
hold on; 
[X, Y] = meshgrid(1:p.grid.nx, 1:p.grid.ny);
quiver(p.grid.dy*Y, p.grid.dx*X, F.x, F.y)


[f, g] = p.evaluate(p.design.z0);
i = 1;
ni_cells = p.nets{i}.cells;
p.plot_heatmap(P); hold on; 
quiver(p.design.x0(ni_cells), p.design.y0(ni_cells), g(ni_cells), g(p.design.n_cells + ni_cells));
hold on; 
p.plot_cells(p.design.x0, p.design.y0, ni_cells)