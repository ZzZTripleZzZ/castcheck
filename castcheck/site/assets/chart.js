/* CastCheck daily-error chart: fetches one API JSON and draws an SVG line. Fails silently;
   every page is complete without it. */
(function () {
  var NS = "http://www.w3.org/2000/svg";
  function e(n, a) { var x = document.createElementNS(NS, n); for (var k in a) x.setAttribute(k, a[k]); return x; }
  function draw(fig, s) {
    var v = s.err_c, d = s.dates, n = v.length, i, y;
    var W = 720, H = 200, L = 40, R = 10, T = 12, B = 22, iw = W - L - R, ih = H - T - B;
    var m = 1;
    for (i = 0; i < n; i++) { y = v[i]; if (y !== null && Math.abs(y) * 1.8 > m) m = Math.abs(y) * 1.8; }
    m = Math.ceil(m);
    var svg = e("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Daily forecast minus observed error in degrees Fahrenheit" });
    function px(i) { return L + (n < 2 ? iw / 2 : i * iw / (n - 1)); }
    function py(f) { return T + ih / 2 - (f / m) * (ih / 2); }
    [m, m / 2, 0, -m / 2, -m].forEach(function (t) {
      var yy = py(t);
      svg.appendChild(e("line", { x1: L, x2: W - R, y1: yy, y2: yy,
        stroke: t === 0 ? "#8a939d" : "#e2e6ea", "stroke-width": 1 }));
      var lb = e("text", { x: L - 6, y: yy + 4, "text-anchor": "end", fill: "#5c6570", "font-size": 10 });
      lb.textContent = (t > 0 ? "+" : "") + (Math.round(t * 10) / 10);
      svg.appendChild(lb);
    });
    var path = "", open = false;
    for (i = 0; i < n; i++) {
      y = v[i];
      if (y === null) { open = false; continue; }
      path += (open ? "L" : "M") + px(i).toFixed(1) + " " + py(y * 1.8).toFixed(1) + " ";
      open = true;
    }
    svg.appendChild(e("path", { d: path, fill: "none", stroke: "#1a5fb4", "stroke-width": 1.6 }));
    for (i = 0; i < n; i++) {
      if (v[i] === null) continue;
      var c = e("circle", { cx: px(i).toFixed(1), cy: py(v[i] * 1.8).toFixed(1), r: 1.9, fill: "#1a5fb4" });
      var ti = e("title"); ti.textContent = d[i] + ": " + (Math.round(v[i] * 18) / 10) + " °F";
      c.appendChild(ti); svg.appendChild(c);
    }
    [[0, d[0]], [n - 1, d[n - 1]]].forEach(function (p, j) {
      var t = e("text", { x: px(p[0]), y: H - 6, "text-anchor": j ? "end" : "start", fill: "#5c6570", "font-size": 10 });
      t.textContent = p[1]; svg.appendChild(t);
    });
    fig.insertBefore(svg, fig.firstChild);
  }
  var figs = document.querySelectorAll("[data-chart]");
  for (var k = 0; k < figs.length; k++) (function (fig) {
    try {
      fetch(fig.getAttribute("data-chart")).then(function (r) { return r.json(); }).then(function (j) {
        var want = { init_hour: +fig.getAttribute("data-init"), method: fig.getAttribute("data-method"),
          variable: fig.getAttribute("data-variable") };
        var ss = (j.series || []).filter(function (s) {
          return s.init_hour === want.init_hour && s.method === want.method && s.variable === want.variable;
        });
        if (ss.length && ss[0].err_c && ss[0].err_c.length) draw(fig, ss[0]);
      })["catch"](function () {});
    } catch (x) {}
  })(figs[k]);
})();
