# Vinet equation of state

Implements the third-order Rose-Vinet EOS used by White & Li (2025) and Boujibar et al. (2020). The pressure is expressed in terms of the Eulerian strain parameter $f = (V/V_0)^{1/3}$ as $P(f) = 3 K_0 f^{-2}(1-f)\exp[\eta(1-f)]$ with $\eta = \tfrac{3}{2}(K_0' - 1)$, where $K_0$ is the zero-pressure isothermal bulk modulus and $K_0'$ its pressure derivative. Inversion to $\rho(P)$ uses Brent root-finding. Material parameters cover Fe (solid: Smith+2018 with the White & Li 2025 light-element correction; liquid: Wicks+2018 Fe-7Si), MgSiO$_3$ perovskite (Fei+2021), MgSiO$_3$ post-perovskite (Sakai+2016), and peridotite (Stixrude & Lithgow-Bertelloni 2005).

::: zalmoxis.eos_vinet
    options:
      inherited_members: false
      show_source: true
      members:
        - get_vinet_density
