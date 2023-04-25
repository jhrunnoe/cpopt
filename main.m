rng('default');

p = Placer();

figure;
p.plot_cells(p.chip.x0, p.chip.y0, []);

while max(p.density, [], "all") > 2
  result = p.solve();
  p.mu = p.mu*3;
  p.sigma = p.mu*10
  p.chip.z0 = result.z;

  figure;
  p.plot_cells(result.z(p.ix), result.z(p.iy), []);
end