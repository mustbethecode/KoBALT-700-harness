/* KoBALT-700 record rendering. Vanilla JS, DOM APIs only, no innerHTML with data.
   One payload (results.json) feeds all pages; every renderer guards on element
   presence, so each page calls only what it contains. */
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var ACCENT = "#0f5b8f";
  var TRACK = "#ececec";
  var GRID = "#c9c9c9";
  var INK = "#1a1a1a";
  var MUTED = "#545454";

  function $(id) { return document.getElementById(id); }

  function el(tag, attrs, text) {
    var node = document.createElement(tag);
    if (attrs) {
      for (var k in attrs) {
        if (Object.prototype.hasOwnProperty.call(attrs, k)) {
          node.setAttribute(k, attrs[k]);
        }
      }
    }
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function svgEl(tag, attrs, text) {
    var node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (var k in attrs) {
        if (Object.prototype.hasOwnProperty.call(attrs, k)) {
          node.setAttribute(k, attrs[k]);
        }
      }
    }
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function isNum(x) { return typeof x === "number" && isFinite(x); }

  function pct(x, digits) {
    if (!isNum(x)) return "not reported";
    return (x * 100).toFixed(digits === undefined ? 1 : digits) + "%";
  }

  function fmtMs(ms) {
    if (!isNum(ms)) return "not recorded";
    if (ms >= 1000) return (ms / 1000).toFixed(1) + " s";
    return Math.round(ms) + " ms";
  }

  function fmtDate(iso) {
    if (typeof iso !== "string") return null;
    var m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
    if (!m) return null;
    var months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    var idx = parseInt(m[2], 10) - 1;
    if (idx < 0 || idx > 11) return null;
    return parseInt(m[3], 10) + " " + months[idx] + " " + m[1];
  }

  function fmtUsd(x, digits) {
    if (!isNum(x)) return "not recorded";
    var d = digits === undefined ? 2 : digits;
    return "$" + x.toFixed(d);
  }

  function costOf(r) {
    return (r && r.cost && isNum(r.cost.usd) && r.cost.usd > 0) ? r.cost.usd : null;
  }

  function costSourceOf(r) {
    return (r && r.cost && typeof r.cost.source === "string") ? r.cost.source : null;
  }

  function costNote(r) {
    var src = costSourceOf(r);
    if (src === "reported") return " (reported)";
    if (src === "measured") return " (measured)";
    return "";
  }

  function evidenceHref(path) {
    if (typeof path !== "string" || !path) return null;
    var p = path.replace(/^\.\//, "");
    if (p.indexOf("site/") === 0) p = p.slice("site/".length);
    return p;
  }

  /* Presentation tiers, decided here at render time. The build only marks
     reasoning effort "none" as ablation; explicit minimal/low effort runs
     arrive as primary and are separated here instead of ranked with
     provider-default runs. */
  function tierOf(r) {
    if (!r) return "disabled";
    if (r.reasoning_mode === "none" || r.category === "ablation") return "disabled";
    if (typeof r.reasoning_mode === "string" && r.reasoning_mode &&
        r.reasoning_mode !== "provider-default") return "constrained";
    return "default";
  }

  function tierShort(t) {
    if (t === "constrained") return "reasoning-limited";
    if (t === "disabled") return "reasoning-disabled";
    return "provider-default";
  }

  function reasoningLabel(run) {
    if (run.reasoning_mode === "provider-default") return "Provider default";
    if (typeof run.reasoning_mode === "string" && run.reasoning_mode) {
      return "Reasoning effort: " + run.reasoning_mode;
    }
    return "Not recorded";
  }

  function shortName(run) {
    if (typeof run.label === "string" && run.label) return run.label;
    if (typeof run.model === "string" && run.model) {
      var parts = run.model.split("/");
      return parts[parts.length - 1];
    }
    return "Unnamed run";
  }

  function modelBase(run) {
    if (typeof run.model === "string" && run.model) {
      var parts = run.model.split("/");
      return parts[parts.length - 1];
    }
    return shortName(run);
  }

  function chartLabel(run) {
    var base = modelBase(run);
    if (tierOf(run) !== "default") base += " (" + run.reasoning_mode + ")";
    return base;
  }

  function setStatus(text, isError) {
    var s = $("status");
    if (!s) return;
    s.textContent = text;
    s.className = isError ? "status error" : "status";
  }

  function chartError(containerId, message) {
    var c = $(containerId);
    if (!c) return;
    while (c.firstChild) c.removeChild(c.firstChild);
    c.appendChild(el("p", { "class": "chart-error" }, message));
    c.removeAttribute("role");
  }

  function evidenceLinks(ev) {
    var p = el("p", null);
    var snap = evidenceHref(ev.snapshot);
    var res = evidenceHref(ev.results);
    if (snap) {
      p.appendChild(el("a", { href: snap }, "Config"));
      if (res) p.appendChild(document.createTextNode(" · "));
    }
    if (res) p.appendChild(el("a", { href: res }, "Results"));
    if (!snap && !res) p.appendChild(el("span", { "class": "small" }, "Evidence unavailable."));
    return p;
  }

  /* ---------------- overall dot-and-interval chart ---------------- */

  function renderOverallChart(runs) {
    var box = $("overall-chart");
    if (!box) return;
    while (box.firstChild) box.removeChild(box.firstChild);
    if (!runs.length) {
      chartError("overall-chart", "No runs to chart.");
      return;
    }
    var W = 680, labelW = 205, valueW = 150, padR = 8, padT = 8, rowH = 40;
    var plotX = labelW, plotW = W - labelW - valueW - padR;
    var H = padT + runs.length * rowH + 26;

    var svg = svgEl("svg", {
      viewBox: "0 0 " + W + " " + H,
      role: "presentation",
      "aria-hidden": "true",
      "font-family": "inherit"
    });
    svg.appendChild(svgEl("desc", null,
      "Dot-and-interval chart. " + runs.map(function (r) {
        var w = r.wilson_95 || {};
        return shortName(r) + ": " + pct(r.accuracy) +
          (isNum(w.lo) && isNum(w.hi)
            ? " (95% interval " + pct(w.lo) + " to " + pct(w.hi) + ")"
            : " (interval unavailable)");
      }).join(". ")));

    [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
      var x = plotX + t * plotW;
      svg.appendChild(svgEl("line", {
        x1: x, y1: padT, x2: x, y2: H - 24, stroke: GRID, "stroke-width": 1
      }));
      svg.appendChild(svgEl("text", {
        x: x, y: H - 8, "text-anchor": "middle",
        "font-size": 11, fill: MUTED
      }, Math.round(t * 100) + "%"));
    });

    runs.forEach(function (r, i) {
      var cy = padT + i * rowH + rowH / 2;
      var name = chartLabel(r);
      var shown = name.length > 28 ? name.slice(0, 27) + "…" : name;
      svg.appendChild(svgEl("text", {
        x: 4, y: cy + 4, "font-size": 12.5, fill: INK
      }, shown));

      var acc = isNum(r.accuracy) ? r.accuracy : null;
      var w = r.wilson_95 || {};
      if (acc !== null && isNum(w.lo) && isNum(w.hi)) {
        var x1 = plotX + Math.max(0, Math.min(1, w.lo)) * plotW;
        var x2 = plotX + Math.max(0, Math.min(1, w.hi)) * plotW;
        svg.appendChild(svgEl("line", {
          x1: x1, y1: cy, x2: x2, y2: cy,
          stroke: INK, "stroke-width": 2, "stroke-linecap": "round"
        }));
        [x1, x2].forEach(function (x) {
          svg.appendChild(svgEl("line", {
            x1: x, y1: cy - 5, x2: x, y2: cy + 5,
            stroke: INK, "stroke-width": 2, "stroke-linecap": "round"
          }));
        });
      }
      if (acc !== null) {
        svg.appendChild(svgEl("circle", {
          cx: plotX + Math.max(0, Math.min(1, acc)) * plotW,
          cy: cy, r: 5.5, fill: ACCENT
        }));
      }
      var val = acc !== null ? pct(acc) : "not reported";
      var ci = (acc !== null && isNum(w.lo) && isNum(w.hi))
        ? " (" + pct(w.lo, 1) + "–" + pct(w.hi, 1) + ")" : "";
      svg.appendChild(svgEl("text", {
        x: W - padR, y: cy + 4, "text-anchor": "end",
        "font-size": 12, fill: INK
      }, val + ci));
    });

    box.appendChild(svg);
  }

  /* ---------------- overview findings ---------------- */

  function levelAcc(run, level) {
    var rows = Array.isArray(run.by_level) ? run.by_level : [];
    for (var i = 0; i < rows.length; i++) {
      if (rows[i] && String(rows[i].name) === String(level)) return rows[i].accuracy;
    }
    return null;
  }

  function renderFindings(tiers) {
    var box = $("findings");
    if (!box) return;
    while (box.firstChild) box.removeChild(box.firstChild);
    var def = tiers.defaultRun || [];
    if (!def.length) {
      box.appendChild(el("li", null, "No provider-default runs in the current data."));
      return;
    }
    var top = def[0];
    var w = top.wilson_95 || {};
    var li1 = el("li", null);
    li1.appendChild(document.createTextNode("The highest observed score is "));
    li1.appendChild(el("strong", null, modelBase(top)));
    li1.appendChild(document.createTextNode(" at " + pct(top.accuracy) +
      (isNum(w.lo) && isNum(w.hi) ? " (95% CI " + pct(w.lo) + "–" + pct(w.hi) + ")" : "") +
      ", " + top.correct + " of " + top.total + ". This is the highest in this set, not a state-of-the-art claim."));
    box.appendChild(li1);

    var l3 = def.map(function (r) { return { r: r, a: levelAcc(r, "1"), b: levelAcc(r, "3") }; })
      .filter(function (x) { return isNum(x.a) && isNum(x.b); });
    if (l3.length) {
      var hi = l3.reduce(function (m, x) { return x.b > m.b ? x : m; });
      var lo = l3.reduce(function (m, x) { return x.b < m.b ? x : m; });
      var allDrop = l3.every(function (x) { return x.b < x.a; });
      var li2 = el("li", null,
        "Difficulty separates the runs: on Level 3 (n=298), scores run from " +
        pct(lo.b) + " (" + modelBase(lo.r) + ") to " + pct(hi.b) + " (" +
        modelBase(hi.r) + ")" + (allDrop ? "; every run scores lowest on Level 3." : "."));
      box.appendChild(li2);
    }

    var pair = findPair(tiers.disabled, def);
    if (pair) {
      var delta = (pair.ab.accuracy - pair.base.accuracy) * 100;
      var li3 = el("li", null,
        "Reasoning matters more than the ranking suggests: switching " +
        modelBase(pair.base) + " from provider-default reasoning to disabled " +
        "changed accuracy from " + pct(pair.base.accuracy) + " to " +
        pct(pair.ab.accuracy) + " (" + (delta > 0 ? "+" : "") + delta.toFixed(1) +
        " points) on the same 700 items. See the paired comparison on the Results page.");
      var a = el("a", { href: "results.html#ablations" }, "Results: reasoning comparisons");
      li3.appendChild(document.createTextNode(" "));
      li3.appendChild(a);
      box.appendChild(li3);
    }
  }

  function findPair(disabled, primary) {
    for (var i = 0; i < disabled.length; i++) {
      var ab = disabled[i];
      if (typeof ab.model !== "string") continue;
      for (var j = 0; j < primary.length; j++) {
        if (primary[j].model === ab.model &&
            isNum(primary[j].accuracy) && isNum(ab.accuracy)) {
          return { base: primary[j], ab: ab };
        }
      }
    }
    return null;
  }

  /* ---------------- tiered comparison tables ---------------- */

  function tierTable(runs, captionText, hideReasoning) {
    var table = el("table", { "class": "compare" });
    table.appendChild(el("caption", null, captionText));
    var thead = el("thead", null);
    var hr = el("tr", null);
    hr.appendChild(el("th", { scope: "col" }, "Model / config"));
    ["Accuracy (95% CI)", "Correct", "Invalid", "Median latency", "Cost (USD)"].forEach(function (h) {
      hr.appendChild(el("th", { scope: "col", "class": "num" }, h));
    });
    // The default tier shares one reasoning setting, shown in the caption
    // instead of a constant column, so the table fits the measure.
    if (!hideReasoning) hr.appendChild(el("th", { scope: "col" }, "Reasoning"));
    hr.appendChild(el("th", { scope: "col" }, "Evidence"));
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = el("tbody", null);
    runs.forEach(function (r) {
      var tr = el("tr", null);
      var th = el("th", { scope: "row" }, modelBase(r));
      // Group membership is stated by the section heading, so no per-row tag:
      // it only widened the first column.
      tr.appendChild(th);
      var w = r.wilson_95 || {};
      tr.appendChild(el("td", { "class": "num soft" },
        pct(r.accuracy) + (isNum(w.lo) && isNum(w.hi)
          ? " (" + pct(w.lo, 1) + "–" + pct(w.hi, 1) + ")" : "")));
      tr.appendChild(el("td", { "class": "num" },
        isNum(r.correct) && isNum(r.total) ? r.correct + " / " + r.total : "n/a"));
      var inv = r.invalid || {};
      tr.appendChild(el("td", { "class": "num" },
        isNum(inv.count) && isNum(r.total)
          ? inv.count + " (" + pct(isNum(inv.rate) ? inv.rate : inv.count / r.total) + ")" : "n/a"));
      tr.appendChild(el("td", { "class": "num" }, fmtMs((r.latency_ms || {}).median)));
      var cost = costOf(r);
      tr.appendChild(el("td", { "class": "num" },
        cost === null ? "not recorded" : fmtUsd(cost) + costNote(r)));
      // In the limited/disabled tables the group heading already states the
      // setting, so the cell carries only the mode word (minimal/low/none).
      if (!hideReasoning) {
        tr.appendChild(el("td", null,
          (typeof r.reasoning_mode === "string" && r.reasoning_mode) ? r.reasoning_mode : "n/a"));
      }
      var tdEv = el("td", null);
      var ev = r.evidence || {};
      var snap = evidenceHref(ev.snapshot);
      var res = evidenceHref(ev.results);
      if (snap) tdEv.appendChild(el("a", { href: snap }, "Config"));
      if (snap && res) tdEv.appendChild(document.createTextNode(" · "));
      if (res) tdEv.appendChild(el("a", { href: res }, "Results"));
      if (!snap && !res) tdEv.appendChild(el("span", { "class": "small" }, "n/a"));
      tr.appendChild(tdEv);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
  }

  function scrollWrap(node, label) {
    var d = el("div", {
      "class": "table-scroll", tabindex: "0", role: "region", "aria-label": label
    });
    d.appendChild(node);
    return d;
  }

  function renderTierTables(tiers) {
    var map = [
      ["tier-default-wrap", tiers.defaultRun, "Provider-default reasoning for every row: same harness, same prompt, provider default reasoning.", true],
      ["tier-constrained-wrap", tiers.constrained, "Reasoning-limited runs (explicit minimal/low effort). Same harness and prompt, but the provider was told to reason less. Not directly comparable to the rows above.", false],
      ["tier-disabled-wrap", tiers.disabled, "Reasoning disabled. Configuration experiments; compare only within the paired panels below, not against the tables above.", false]
    ];
    map.forEach(function (m) {
      var wrap = $(m[0]);
      if (!wrap) return;
      while (wrap.firstChild) wrap.removeChild(wrap.firstChild);
      if (!m[1].length) {
        wrap.appendChild(el("p", { "class": "small" }, "No runs in this group in the current data."));
        return;
      }
      wrap.appendChild(scrollWrap(tierTable(m[1], m[2], m[3]), m[2].split(".")[0]));
    });
  }

  /* ---------------- breakdown charts and tables ---------------- */

  function breakdownGroups(runs, key) {
    var names = [];
    runs.forEach(function (r) {
      (Array.isArray(r[key]) ? r[key] : []).forEach(function (g) {
        if (g && typeof g.name === "string" && names.indexOf(g.name) === -1) names.push(g.name);
      });
    });
    return names;
  }

  function renderBreakdownChart(containerId, runs, key, groupLabel) {
    var box = $(containerId);
    if (!box) return false;
    while (box.firstChild) box.removeChild(box.firstChild);
    var names = breakdownGroups(runs, key);
    if (!runs.length || !names.length) {
      chartError(containerId, "Breakdown data unavailable.");
      return false;
    }
    var W = 680, labelW = 205, valueW = 56, padR = 10;
    var plotX = labelW, plotW = W - labelW - valueW - padR;
    var rowH = 24, blockTitleH = 28, blockGap = 12, padT = 8, padB = 8;
    var H = padT + runs.length * (blockTitleH + names.length * rowH + blockGap) + padB;

    var svg = svgEl("svg", {
      viewBox: "0 0 " + W + " " + H,
      role: "presentation", "aria-hidden": "true", "font-family": "inherit"
    });
    svg.appendChild(svgEl("desc", null,
      "Bar chart of " + groupLabel + " accuracy by run, zero to one hundred percent."));

    var y = padT;
    runs.forEach(function (r) {
      var title = chartLabel(r);
      if (title.length > 50) title = title.slice(0, 49) + "…";
      svg.appendChild(svgEl("text", {
        x: 4, y: y + 15, "font-size": 13, "font-weight": "bold", fill: INK
      }, title));
      y += blockTitleH;
      var byName = {};
      (Array.isArray(r[key]) ? r[key] : []).forEach(function (g) {
        if (g && typeof g.name === "string") byName[g.name] = g;
      });
      names.forEach(function (n) {
        var g = byName[n] || {};
        var label = n + (isNum(g.n) ? " (n=" + g.n + ")" : "");
        if (label.length > 28) label = label.slice(0, 27) + "…";
        svg.appendChild(svgEl("text", {
          x: 4, y: y + 15, "font-size": 12, fill: MUTED
        }, label));
        var acc = isNum(g.accuracy) ? Math.max(0, Math.min(1, g.accuracy)) : null;
        svg.appendChild(svgEl("rect", {
          x: plotX, y: y + 4, width: plotW, height: 13, fill: TRACK
        }));
        if (acc !== null) {
          svg.appendChild(svgEl("rect", {
            x: plotX, y: y + 4, width: Math.max(acc * plotW, 2), height: 13, fill: ACCENT
          }));
          svg.appendChild(svgEl("text", {
            x: plotX + plotW + 5, y: y + 15, "font-size": 12, fill: INK
          }, pct(g.accuracy)));
        } else {
          svg.appendChild(svgEl("text", {
            x: plotX + plotW + 5, y: y + 15, "font-size": 12, fill: MUTED
          }, "n/a"));
        }
        y += rowH;
      });
      y += blockGap;
    });

    box.appendChild(svg);
    return true;
  }

  function renderBreakdownTable(wrapId, runs, key, firstColLabel, tagTiers) {
    var wrap = $(wrapId);
    if (!wrap) return;
    while (wrap.firstChild) wrap.removeChild(wrap.firstChild);
    var names = breakdownGroups(runs, key);
    if (!runs.length || !names.length) {
      wrap.appendChild(el("p", { "class": "small" }, "Breakdown data unavailable."));
      return;
    }
    var table = el("table", null);
    table.appendChild(el("caption", null,
      "Exact " + firstColLabel.toLowerCase() + " accuracy: correct / total (percent)."));
    var thead = el("thead", null);
    var hr = el("tr", null);
    hr.appendChild(el("th", { scope: "col" }, firstColLabel));
    runs.forEach(function (r) {
      var h = modelBase(r) + (tagTiers && tierOf(r) !== "default" ? " [" + tierShort(tierOf(r)) + "]" : "");
      hr.appendChild(el("th", { scope: "col" }, h));
    });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = el("tbody", null);
    names.forEach(function (n) {
      var tr = el("tr", null);
      var n0 = null;
      runs.forEach(function (r) {
        var gs = (Array.isArray(r[key]) ? r[key] : []).filter(function (x) {
          return x && x.name === n;
        });
        if (gs.length && isNum(gs[0].n) && n0 === null) n0 = gs[0].n;
      });
      tr.appendChild(el("th", { scope: "row" }, n + (n0 !== null ? " (n=" + n0 + ")" : "")));
      runs.forEach(function (r) {
        var gs = (Array.isArray(r[key]) ? r[key] : []).filter(function (x) {
          return x && x.name === n;
        });
        var g = gs.length ? gs[0] : null;
        tr.appendChild(el("td", { "class": "num" },
          (!g || !isNum(g.correct) || !isNum(g.n))
            ? "n/a" : g.correct + " / " + g.n + " (" + pct(g.accuracy) + ")"));
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(scrollWrap(table, firstColLabel + " values by run"));
  }

  /* ---------------- cost: accuracy vs run cost ---------------- */

  function pricedRuns(runs) {
    return runs.filter(function (r) { return costOf(r) !== null; });
  }

  function renderCostChart(runs) {
    var box = $("cost-chart");
    if (!box) return false;
    while (box.firstChild) box.removeChild(box.firstChild);
    var priced = pricedRuns(runs);
    if (!priced.length) {
      chartError("cost-chart", "No cost recorded for provider-default runs.");
      return false;
    }
    var W = 680, H = 380, padL = 58, padR = 24, padT = 14, padB = 52;
    var plotW = W - padL - padR, plotH = H - padT - padB;
    var minC = Infinity, maxC = 0;
    priced.forEach(function (r) {
      var c = costOf(r);
      if (c < minC) minC = c;
      if (c > maxC) maxC = c;
    });
    var lo = minC / 1.25, hi = maxC * 1.25;
    function xFor(c) {
      return padL + (Math.log(c) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * plotW;
    }
    function yFor(a) {
      return padT + (1 - Math.max(0, Math.min(1, a))) * plotH;
    }

    var svg = svgEl("svg", {
      viewBox: "0 0 " + W + " " + H,
      role: "presentation", "aria-hidden": "true", "font-family": "inherit"
    });
    svg.appendChild(svgEl("desc", null,
      "Scatter chart of accuracy against run cost in US dollars, log scale. " +
      priced.map(function (r) {
        var w = r.wilson_95 || {};
        return shortName(r) + ": " + pct(r.accuracy) + " at " + fmtUsd(costOf(r)) +
          " (" + (costSourceOf(r) || "source not recorded") + ")" +
          (isNum(w.lo) && isNum(w.hi)
            ? ", 95% interval " + pct(w.lo) + " to " + pct(w.hi) : "");
      }).join(". ")));

    [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
      var y = yFor(t);
      svg.appendChild(svgEl("line", {
        x1: padL, y1: y, x2: padL + plotW, y2: y, stroke: GRID, "stroke-width": 1
      }));
      svg.appendChild(svgEl("text", {
        x: padL - 6, y: y + 4, "text-anchor": "end", "font-size": 11, fill: MUTED
      }, Math.round(t * 100) + "%"));
    });

    var ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].filter(function (v) {
      return v >= lo && v <= hi;
    });
    if (!ticks.length) ticks = [lo, hi];
    ticks.forEach(function (v) {
      var x = xFor(v);
      svg.appendChild(svgEl("line", {
        x1: x, y1: padT, x2: x, y2: padT + plotH, stroke: GRID, "stroke-width": 1
      }));
      svg.appendChild(svgEl("text", {
        x: x, y: padT + plotH + 16, "text-anchor": "middle", "font-size": 11, fill: MUTED
      }, "$" + (v >= 1 ? String(v) : v.toFixed(2))));
    });
    svg.appendChild(svgEl("text", {
      x: padL + plotW / 2, y: H - 8, "text-anchor": "middle", "font-size": 11, fill: MUTED
    }, "Run cost (USD, log scale)"));

    priced.forEach(function (r, i) {
      var c = costOf(r), acc = isNum(r.accuracy) ? r.accuracy : null;
      if (acc === null) return;
      var x = xFor(c), y = yFor(acc);
      var w = r.wilson_95 || {};
      if (isNum(w.lo) && isNum(w.hi)) {
        svg.appendChild(svgEl("line", {
          x1: x, y1: yFor(w.hi), x2: x, y2: yFor(w.lo),
          stroke: INK, "stroke-width": 2, "stroke-linecap": "round"
        }));
        [yFor(w.lo), yFor(w.hi)].forEach(function (yy) {
          svg.appendChild(svgEl("line", {
            x1: x - 5, y1: yy, x2: x + 5, y2: yy,
            stroke: INK, "stroke-width": 2, "stroke-linecap": "round"
          }));
        });
      }
      // Filled marker: measured provider cost. Hollow: operator-reported.
      svg.appendChild(svgEl("circle", costSourceOf(r) === "measured"
        ? { cx: x, cy: y, r: 5.5, fill: ACCENT }
        : { cx: x, cy: y, r: 5.5, fill: "#ffffff", stroke: ACCENT, "stroke-width": 2 }));
      var label = modelBase(r);
      if (label.length > 22) label = label.slice(0, 21) + "…";
      // Right-edge points carry their label on the left so it never overflows;
      // alternating vertical offsets keep neighbouring labels apart.
      var toLeft = x > padL + plotW * 0.55;
      var labelY = y + 4 + (i % 2 === 0 ? -14 : 14);
      svg.appendChild(svgEl("text", {
        x: toLeft ? x - 9 : x + 9, y: labelY,
        "text-anchor": toLeft ? "end" : "start",
        "font-size": 12, fill: INK
      }, label + " — " + fmtUsd(c) + costNote(r)));
    });

    box.appendChild(svg);
    return true;
  }

  function renderCostTable(wrapId, runs) {
    var wrap = $(wrapId);
    if (!wrap) return;
    while (wrap.firstChild) wrap.removeChild(wrap.firstChild);
    var priced = pricedRuns(runs);
    if (!priced.length) {
      wrap.appendChild(el("p", { "class": "small" }, "No cost recorded for these runs."));
      return;
    }
    var table = el("table", null);
    table.appendChild(el("caption", null,
      "Exact cost: total run cost in USD, its source, cost per correct item, and accuracy."));
    var thead = el("thead", null);
    var hr = el("tr", null);
    ["Run", "Cost (USD)", "Source", "$ per correct", "Accuracy (95% CI)"].forEach(function (h) {
      hr.appendChild(el("th", { scope: "col" }, h));
    });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = el("tbody", null);
    priced.forEach(function (r) {
      var tr = el("tr", null);
      var t = tierOf(r);
      tr.appendChild(el("th", { scope: "row" },
        modelBase(r) + (t !== "default" ? " [" + tierShort(t) + "]" : "")));
      tr.appendChild(el("td", { "class": "num" }, fmtUsd(costOf(r))));
      tr.appendChild(el("td", null, costSourceOf(r) || "not recorded"));
      tr.appendChild(el("td", { "class": "num" },
        (r.cost && isNum(r.cost.per_correct)) ? fmtUsd(r.cost.per_correct, 4) : "n/a"));
      var w = r.wilson_95 || {};
      tr.appendChild(el("td", { "class": "num" }, pct(r.accuracy) +
        (isNum(w.lo) && isNum(w.hi)
          ? " (" + pct(w.lo, 1) + "–" + pct(w.hi, 1) + ")" : "")));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(scrollWrap(table, "Exact cost values by run"));
  }

  /* ---------------- reasoning comparisons ---------------- */

  function renderAblations(disabled, primary) {
    var body = $("ablation-body");
    if (!body) return;
    while (body.firstChild) body.removeChild(body.firstChild);
    if (!disabled.length) {
      body.appendChild(el("p", null, "No reasoning-disabled runs in the current data."));
      return;
    }
    // Paired first, unpaired observations after.
    var paired = [], unpaired = [];
    disabled.forEach(function (ab) {
      var found = null;
      if (typeof ab.model === "string") {
        for (var i = 0; i < primary.length; i++) {
          if (primary[i].model === ab.model) { found = primary[i]; break; }
        }
      }
      (found ? paired : unpaired).push({ ab: ab, base: found });
    });
    paired.concat(unpaired).forEach(function (item) {
      var ab = item.ab, base = item.base;
      var panel = el("div", { "class": "ablation" });
      panel.appendChild(el("h3", null, shortName(ab)));
      if (base && isNum(base.accuracy) && isNum(ab.accuracy)) {
        var delta = (ab.accuracy - base.accuracy) * 100;
        panel.appendChild(el("p", { "class": "delta" },
          pct(base.accuracy) + " → " + pct(ab.accuracy) +
          "  (" + (delta > 0 ? "+" : "") + delta.toFixed(1) + " points)"));
        panel.appendChild(el("p", null,
          "Same model, same 700 items; only the provider reasoning setting changed " +
          "(provider default → disabled). " +
          (delta <= 0 ? "Disabling reasoning lowered" : "Disabling reasoning raised") +
          " accuracy by " + Math.abs(delta).toFixed(1) +
          " points. A configuration result, not a model rank."));
        var bits = [];
        var latB = (base.latency_ms || {}).median, latA = (ab.latency_ms || {}).median;
        if (isNum(latB) && isNum(latA)) bits.push("median latency " + fmtMs(latB) + " → " + fmtMs(latA));
        var invB = (base.invalid || {}).count, invA = (ab.invalid || {}).count;
        if (isNum(invB) && isNum(invA) && isNum(base.total)) {
          bits.push("invalid outputs " + invB + " → " + invA + " of " + base.total);
        }
        if (bits.length) panel.appendChild(el("p", { "class": "ablation-note" }, bits.join("; ") + "."));
      } else {
        panel.appendChild(el("p", null,
          "Accuracy " + pct(ab.accuracy) + " with reasoning disabled (" +
          (isNum(ab.correct) ? ab.correct + " of " + ab.total : "counts unavailable") + "). " +
          "There is no provider-default run of this model in the current data, so this " +
          "stands as an unpaired observation: it cannot show what disabling reasoning " +
          "changed, only where this configuration landed."));
      }
      panel.appendChild(evidenceLinks(ab.evidence || {}));
      body.appendChild(panel);
    });
  }

  /* ---------------- reliability ---------------- */

  function renderReliability(allRuns) {
    var wrap = $("reliability-table-wrap");
    if (!wrap) return;
    while (wrap.firstChild) wrap.removeChild(wrap.firstChild);
    if (!allRuns.length) {
      wrap.appendChild(el("p", { "class": "small" }, "Reliability data unavailable."));
      return;
    }
    var table = el("table", null);
    table.appendChild(el("caption", null,
      "Invalid outputs and latency per run. Latency reflects the endpoint used, not the model itself."));
    var thead = el("thead", null);
    var hr = el("tr", null);
    hr.appendChild(el("th", { scope: "col" }, "Run"));
    ["Invalid", "Empty outputs", "Median latency", "p95 latency"].forEach(function (h) {
      hr.appendChild(el("th", { scope: "col", "class": "num" }, h));
    });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = el("tbody", null);
    allRuns.forEach(function (r) {
      var tr = el("tr", null);
      var t = tierOf(r);
      tr.appendChild(el("th", { scope: "row" },
        modelBase(r) + (t !== "default" ? " [" + tierShort(t) + "]" : "")));
      var inv = r.invalid || {};
      tr.appendChild(el("td", { "class": "num" },
        isNum(inv.count) && isNum(r.total)
          ? inv.count + " / " + r.total + " (" + pct(isNum(inv.rate) ? inv.rate : inv.count / r.total) + ")"
          : "n/a"));
      tr.appendChild(el("td", { "class": "num" },
        (typeof r.empty_raw_output === "number" && isNum(r.total))
          ? r.empty_raw_output + " / " + r.total : "n/a"));
      var lat = r.latency_ms || {};
      tr.appendChild(el("td", { "class": "num" }, fmtMs(lat.median)));
      tr.appendChild(el("td", { "class": "num" }, fmtMs(lat.p95)));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(scrollWrap(table, "Invalid outputs and latency by run"));
  }

  /* ---------------- runs archive ---------------- */

  function renderRunIndex(allRuns) {
    var wrap = $("run-index-wrap");
    if (!wrap) return;
    while (wrap.firstChild) wrap.removeChild(wrap.firstChild);
    if (!allRuns.length) {
      wrap.appendChild(el("p", { "class": "small" }, "No runs in the current data."));
      return;
    }
    var table = el("table", null);
    table.appendChild(el("caption", null, "All configurations. Detail follows below, one section per run."));
    var thead = el("thead", null);
    var hr = el("tr", null);
    hr.appendChild(el("th", { scope: "col" }, "Run"));
    hr.appendChild(el("th", { scope: "col", "class": "num" }, "Accuracy"));
    hr.appendChild(el("th", { scope: "col", "class": "num" }, "Cost (USD)"));
    hr.appendChild(el("th", { scope: "col" }, "Reasoning"));
    hr.appendChild(el("th", { scope: "col" }, "Run date"));
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = el("tbody", null);
    allRuns.forEach(function (r) {
      var tr = el("tr", null);
      var th = el("th", { scope: "row" });
      th.appendChild(el("a", { href: "#" + r.slug }, modelBase(r)));
      var t = tierOf(r);
      if (t !== "default") th.appendChild(el("div", { "class": "run-id" }, tierShort(t)));
      tr.appendChild(th);
      tr.appendChild(el("td", { "class": "num" }, pct(r.accuracy)));
      var idxCost = costOf(r);
      tr.appendChild(el("td", { "class": "num" },
        idxCost === null ? "not recorded" : fmtUsd(idxCost) + costNote(r)));
      tr.appendChild(el("td", null, reasoningLabel(r)));
      var d = typeof r.timestamp === "string" ? fmtDate(r.timestamp) : null;
      tr.appendChild(el("td", null, d || "not recorded"));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(scrollWrap(table, "Index of runs"));
  }

  function renderArchive(allRuns) {
    var box = $("run-archive");
    if (!box) return;
    while (box.firstChild) box.removeChild(box.firstChild);
    allRuns.forEach(function (r) {
      var sec = el("section", { "class": "run-entry", id: r.slug });
      sec.appendChild(el("h2", null, shortName(r)));
      if (typeof r.model === "string" && r.model) {
        sec.appendChild(el("p", { "class": "run-id" }, "Model id: " + r.model));
      }
      var dl = el("dl", { "class": "facts" });
      function row(term, value) {
        dl.appendChild(el("dt", null, term));
        dl.appendChild(el("dd", null, value));
      }
      var w = r.wilson_95 || {};
      row("Accuracy", pct(r.accuracy) +
        (isNum(w.lo) && isNum(w.hi) ? " (95% CI " + pct(w.lo) + "–" + pct(w.hi) + ")" : ""));
      row("Correct", isNum(r.correct) && isNum(r.total) ? r.correct + " / " + r.total : "n/a");
      var inv = r.invalid || {};
      row("Invalid outputs", isNum(inv.count) && isNum(r.total)
        ? inv.count + " (" + pct(isNum(inv.rate) ? inv.rate : inv.count / r.total) + ")" : "n/a");
      var lat = r.latency_ms || {};
      row("Latency (median / p95)", fmtMs(lat.median) + " / " + fmtMs(lat.p95));
      var arcCost = costOf(r);
      row("Cost", arcCost === null
        ? "not recorded"
        : fmtUsd(arcCost) + " (" + (costSourceOf(r) || "source not recorded") + ")" +
          (r.cost && isNum(r.cost.per_correct)
            ? " · " + fmtUsd(r.cost.per_correct, 4) + " per correct item" : ""));
      if (r.usage && isNum(r.usage.prompt_tokens) && isNum(r.usage.completion_tokens)) {
        var rt = isNum(r.usage.reasoning_tokens) ? " / " + r.usage.reasoning_tokens : "";
        row("Tokens (prompt / completion" + (rt ? " / reasoning" : "") + ")",
          r.usage.prompt_tokens + " / " + r.usage.completion_tokens + rt);
      }
      row("Reasoning", reasoningLabel(r) + " — " + tierShort(tierOf(r)) + " group");
      row("Endpoint", typeof r.endpoint === "string" && r.endpoint ? r.endpoint : "not recorded");
      row("Max tokens", r.max_new_tokens !== undefined && r.max_new_tokens !== null
        ? String(r.max_new_tokens) : "not recorded");
      var d = typeof r.timestamp === "string" ? fmtDate(r.timestamp) : null;
      row("Run date", d ? d + " (UTC)" : "not recorded");
      if (typeof r.prompt_template_hash === "string" && r.prompt_template_hash) {
        row("Prompt hash", r.prompt_template_hash.slice(0, 12) + "…");
      }
      sec.appendChild(dl);
      sec.appendChild(evidenceLinks(r.evidence || {}));
      box.appendChild(sec);
    });
  }

  /* ---------------- skipped ---------------- */

  function renderSkipped(skipped) {
    var body = $("skipped-body");
    var summary = $("skipped-summary");
    if (!body || !summary) return;
    while (body.firstChild) body.removeChild(body.firstChild);
    if (!skipped || !skipped.length) {
      summary.textContent = "Skipped or incomplete runs (0)";
      body.appendChild(el("p", { "class": "small" }, "No runs were skipped."));
      return;
    }
    summary.textContent = "Skipped or incomplete runs (" + skipped.length + ")";
    var ul = el("ul", { "class": "tight" });
    skipped.forEach(function (s) {
      var slug = (s && typeof s.slug === "string") ? s.slug : "unknown run";
      var reason = (s && typeof s.reason === "string") ? s.reason : "reason not recorded";
      ul.appendChild(el("li", null,
        slug + " — " + reason + ". Excluded from all results and charts."));
    });
    body.appendChild(ul);
  }

  /* ---------------- meta ---------------- */

  function renderMeta(payload) {
    var runs = Array.isArray(payload.runs) ? payload.runs : [];
    var foot = $("footer-line");
    if (foot) {
      var gen = typeof payload.generated_at === "string" ? fmtDate(payload.generated_at) : null;
      foot.textContent = "KoBALT-700 benchmark record." +
        (gen ? " Data generated " + gen + " (UTC)" : "") +
        " from run directories via site/build.py.";
    }
    var meta = $("report-meta");
    if (!meta) return;
    while (meta.firstChild) meta.removeChild(meta.firstChild);
    var stamps = runs
      .map(function (r) { return typeof r.timestamp === "string" ? r.timestamp : null; })
      .filter(Boolean).sort();
    var nDef = runs.filter(function (r) { return tierOf(r) === "default"; }).length;
    var parts = [];
    if (stamps.length) {
      var first = fmtDate(stamps[0]), last = fmtDate(stamps[stamps.length - 1]);
      if (first) parts.push(first === last || !last ? first : first + " – " + last);
    }
    if (isNum(payload.item_count)) parts.push(payload.item_count + " items per run");
    parts.push(runs.length + " configurations (" + nDef + " provider-default)");
    meta.textContent = parts.join(" · ") + ".";
  }

  /* ---------------- main ---------------- */

  // Deferred reveal (CLS guard): dynamic regions are display:none until this
  // runs, so the whole data render lands in a single frame below already
  // painted static content. Nothing previously painted moves, which keeps
  // Cumulative Layout Shift near zero. Always called exactly once per load,
  // on success and on failure alike, so content is never hidden indefinitely.
  function reveal() {
    document.body.classList.add("is-ready");
    var main = $("main");
    if (main) main.removeAttribute("aria-busy");
    // Restore deep links (e.g. runs.html#<slug>): the target was hidden at
    // parse time, so re-resolve the fragment after reveal.
    try {
      if (typeof window.location.hash === "string" && window.location.hash.length > 1) {
        var t = document.getElementById(window.location.hash.slice(1));
        if (t && typeof t.scrollIntoView === "function") t.scrollIntoView();
      }
    } catch (e) { /* scrolling is best-effort */ }
  }

  function fail(message) {
    setStatus(message, true);
    ["overall-chart", "domain-chart", "level-chart", "cost-chart"].forEach(function (id) {
      chartError(id, "Unavailable: " + message);
    });
    var box = $("findings");
    if (box) {
      while (box.firstChild) box.removeChild(box.firstChild);
      box.appendChild(el("li", null, "Findings unavailable: " + message));
    }
    renderSkipped([]);
    reveal();
  }

  function init() {
    var main = $("main");
    if (main) main.setAttribute("aria-busy", "true");
    fetch("results.json", { cache: "no-store" })
      .then(function (resp) {
        if (!resp.ok) throw new Error("results.json returned HTTP " + resp.status);
        return resp.json();
      })
      .then(function (payload) {
        if (!payload || typeof payload !== "object" || !Array.isArray(payload.runs)) {
          fail("results file is missing the runs list.");
          return;
        }
        var runs = payload.runs.filter(Boolean);
        if (!runs.length) {
          fail("no completed runs found in the results file.");
          return;
        }
        var tiers = {
          defaultRun: runs.filter(function (r) { return tierOf(r) === "default"; }),
          constrained: runs.filter(function (r) { return tierOf(r) === "constrained"; }),
          disabled: runs.filter(function (r) { return tierOf(r) === "disabled"; })
        };
        var allPrimary = tiers.defaultRun.concat(tiers.constrained);

        renderMeta(payload);
        renderOverallChart(tiers.defaultRun);
        renderFindings(tiers);
        renderTierTables(tiers);
        var okC = renderCostChart(tiers.defaultRun);
        renderCostTable("cost-table-wrap", tiers.defaultRun);
        var chartRuns = tiers.defaultRun;
        var tableRuns = allPrimary;
        var okD = renderBreakdownChart("domain-chart", chartRuns, "by_domain", "domain");
        renderBreakdownTable("domain-table-wrap", tableRuns, "by_domain", "Domain", true);
        var okL = renderBreakdownChart("level-chart", chartRuns, "by_level", "difficulty level");
        renderBreakdownTable("level-table-wrap", tableRuns, "by_level", "Level", true);
        renderAblations(tiers.disabled, allPrimary);
        renderReliability(runs);
        renderRunIndex(runs);
        renderArchive(runs);
        renderSkipped(payload.skipped);

        var notes = [];
        if ($("domain-chart") && !okD) notes.push("domain chart unavailable");
        if ($("level-chart") && !okL) notes.push("difficulty chart unavailable");
        if ($("cost-chart") && !okC) notes.push("cost chart unavailable");
        setStatus("Loaded " + runs.length + " configurations (" +
          tiers.defaultRun.length + " provider-default, " +
          tiers.constrained.length + " reasoning-limited, " +
          tiers.disabled.length + " reasoning-disabled)." +
          (notes.length ? " Note: " + notes.join("; ") + "." : ""));
        reveal();
      })
      .catch(function (err) {
        fail(err && err.message ? err.message : "could not load results.json.");
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
