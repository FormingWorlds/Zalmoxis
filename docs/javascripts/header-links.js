console.log("header-links.js loaded");

function wire() {
  const homepage = "https://proteus-framework.org/";

  const logo = document.querySelector(".md-header__button.md-logo");
  if (logo) logo.href = homepage;

  const title = document.querySelector(".md-header__title[data-md-component='header-title']");
  if (title && !title.dataset.titleWired) {
    title.dataset.titleWired = "1";
    title.style.cursor = "pointer";

    // always go to /Zalmoxis/ when hosted there, else "/" (mkdocs serve)
    const href = location.href;
    const docsHome = href.includes("/Zalmoxis/")
      ? href.split("/Zalmoxis/")[0] + "/Zalmoxis/"
      : location.origin + "/";

    title.addEventListener("click", (e) => {
      if (e.target.closest("a, button, input, label")) return;
      window.location.assign(docsHome);
    }, true);
  }
}

document.addEventListener("DOMContentLoaded", wire);
if (window.document$?.subscribe) window.document$.subscribe(wire);
