(function () {
  const OPEN = new Set(["Reference", "Community", "Other PROTEUS modules"]);

  function textOf(el) {
    return (el && el.textContent ? el.textContent : "").trim();
  }

  function expandActiveBranch(sidebar) {
    // Active link can be on <a> or sometimes on the <li>
    const active =
      sidebar.querySelector(".md-nav__link--active") ||
      sidebar.querySelector(".md-nav__item--active > .md-nav__link");

    if (!active) return;

    // Walk up through parent <li> nodes and check their toggles
    let li = active.closest("li");
    while (li && li !== sidebar) {
      const toggle = li.querySelector(":scope > input.md-nav__toggle[type='checkbox']");
      if (toggle) toggle.checked = true;
      li = li.parentElement ? li.parentElement.closest("li") : null;
    }
  }

  function applyDefaultSidebarState() {
    const sidebar = document.querySelector(".md-sidebar--primary");
    if (!sidebar) return;

    // Collapse everything first
    sidebar.querySelectorAll('input.md-nav__toggle[type="checkbox"]').forEach(cb => {
      cb.checked = false;
    });

    // Expand top-level sections (Homepage defaults)
    const items = sidebar.querySelectorAll(".md-nav__item--nested");
    items.forEach(item => {
      const titleEl = item.querySelector(":scope > label.md-nav__link, :scope > a.md-nav__link");
      const title = textOf(titleEl);

      if (OPEN.has(title)) {
        const toggle = item.querySelector(":scope > input.md-nav__toggle[type='checkbox']");
        if (toggle) toggle.checked = true;
      }
    });

    // Finally, ALSO expand whatever branch contains the current page
    // (so "API reference" opens when you're on its children)
    expandActiveBranch(sidebar);
  }

  document.addEventListener("DOMContentLoaded", applyDefaultSidebarState);
  document.addEventListener("navigation:complete", applyDefaultSidebarState);
})();