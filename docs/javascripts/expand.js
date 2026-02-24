(function () {
  const OPEN = new Set(["Reference", "Community", "Other PROTEUS modules"]);

  function textOf(el) {
    return (el && el.textContent ? el.textContent : "").trim();
  }

  function applyDefaultSidebarState() {
    const sidebar = document.querySelector(".md-sidebar--primary");
    if (!sidebar) return;

    // collapse all
    sidebar.querySelectorAll('input.md-nav__toggle[type="checkbox"]').forEach(cb => {
      cb.checked = false;
    });

    // expand only selected
    const items = sidebar.querySelectorAll(".md-nav__item--nested");
    items.forEach(item => {
      // In Material, the visible title can be either an <a> or a <label>
      const titleEl = item.querySelector(":scope > label.md-nav__link, :scope > a.md-nav__link");
      const title = textOf(titleEl);

      if (OPEN.has(title)) {
        const toggle = item.querySelector(":scope > input.md-nav__toggle[type='checkbox']");
        if (toggle) toggle.checked = true;
      }
    });
  }

  document.addEventListener("DOMContentLoaded", applyDefaultSidebarState);
  document.addEventListener("navigation:complete", applyDefaultSidebarState);
})();