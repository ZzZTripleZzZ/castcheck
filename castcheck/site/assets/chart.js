/* CastCheck progressive enhancement. Everything on the page — tables and SVG alike — is written
   by the static build; this file adds only the theme toggle and a hover read-out. Deleting it
   loses no information. */
(function () {
  "use strict";

  /* ---- theme toggle: system → light → dark → system, remembered per browser ---- */
  var btn = document.getElementById("theme-toggle");
  if (btn) {
    var LABEL = { system: "Theme: system", light: "Theme: light", dark: "Theme: dark" };
    var stored = null;
    try { stored = localStorage.getItem("castcheck-theme"); } catch (e) {}
    var mode = stored === "light" || stored === "dark" ? stored : "system";
    function apply(m) {
      mode = m;
      if (m === "system") {
        document.documentElement.removeAttribute("data-theme");
        try { localStorage.removeItem("castcheck-theme"); } catch (e) {}
      } else {
        document.documentElement.setAttribute("data-theme", m);
        try { localStorage.setItem("castcheck-theme", m); } catch (e) {}
      }
      btn.textContent = LABEL[m];
      btn.setAttribute("aria-pressed", m === "dark" ? "true" : "false");
    }
    btn.hidden = false;
    apply(mode);
    btn.addEventListener("click", function () {
      apply(mode === "system" ? "light" : mode === "light" ? "dark" : "system");
    });
  }

  /* ---- hover read-out: the SVG points already carry <title>, this makes them legible fast ---- */
  var figs = document.querySelectorAll(".fig");
  for (var i = 0; i < figs.length; i++) (function (fig) {
    var out = fig.querySelector(".readout");
    if (!out) return;
    var idle = out.textContent;
    fig.addEventListener("mouseover", function (ev) {
      var t = ev.target;
      if (!t || !t.getAttribute) return;
      var title = t.querySelector && t.querySelector("title");
      if (title) out.textContent = title.textContent;
    });
    fig.addEventListener("mouseleave", function () { out.textContent = idle; });
  })(figs[i]);
})();
