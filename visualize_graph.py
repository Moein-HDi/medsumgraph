"""Generate an interactive HTML explorer for the knowledge graph.

Usage: python visualize_graph.py [--out graph_viz.html] [--max-nodes N]
"""
import argparse
import json
from collections import Counter

import config


def load_graph() -> dict:
    with open(config.GRAPH_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_html(graph: dict, max_nodes: int) -> str:
    edges = graph["edges"]
    triples = [(s, p, o) for s, outs in edges.items() for p, o in outs]

    degree = Counter()
    for s, outs in edges.items():
        degree[s] += len(outs)
        for _, o in outs:
            degree[o] += 1

    # keep the most-connected nodes if too dense
    if max_nodes and len(degree) > max_nodes:
        keep = set(n for n, _ in degree.most_common(max_nodes))
        triples = [(s, p, o) for s, p, o in triples if s in keep and o in keep]

    nodes_set = sorted({s for s, _, _ in triples} | {o for _, _, o in triples})
    node_ids = {n: i for i, n in enumerate(nodes_set)}
    links = [{"source": node_ids[s], "target": node_ids[o], "pred": p} for s, p, o in triples]

    # color by degree: hot = high degree
    node_colors = []
    max_deg = max((degree.get(n, 0) for n in nodes_set), default=1)
    for n in nodes_set:
        d = degree.get(n, 0)
        t = d / max_deg
        r = int(180 + 75 * t)
        b = int(230 - 130 * t)
        node_colors.append(f"rgb({r},150,{b})")

    payload = {
        "nodes": nodes_set,
        "links": links,
        "colors": node_colors,
        "degree": [degree.get(n, 0) for n in nodes_set],
    }
    data = json.dumps(payload, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MedSumGraph — Knowledge Graph Explorer</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  :root {{
    --bg: #0b1220;
    --panel-bg: rgba(13, 20, 36, 0.97);
    --border: #1e2a44;
    --text: #e8eefc;
    --muted: #8aa0c8;
    --accent: #4da3ff;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: 'Segoe UI', system-ui, sans-serif;
         background: var(--bg); color: var(--text); overflow: hidden; }}
  #toolbar {{ position: fixed; top: 0; left: 0; right: 0; z-index: 10;
             background: rgba(11,18,32,0.92); padding: 10px 16px;
             display: flex; gap: 12px; align-items: center;
             border-bottom: 1px solid var(--border); backdrop-filter: blur(4px); }}
  #toolbar b {{ color: #fff; letter-spacing: 0.4px; }}
  #search {{ width: 280px; padding: 7px 12px; border-radius: 6px;
            border: 1px solid #2a3a5e; background: #101a30; color: #f0f4ff;
            font-size: 13px; outline: none; }}
  #search:focus {{ border-color: var(--accent); }}
  #toolbar button {{ padding: 7px 14px; border-radius: 6px; border: 1px solid #2a3a5e;
                     background: #131e36; color: #dbe6ff; cursor: pointer; font-size: 13px; }}
  #toolbar button:hover {{ background: #1b2a4a; }}
  #stats {{ margin-left: auto; font-size: 12px; color: var(--muted); }}
  svg {{ display: block; }}
  .link {{ stroke: #5b6f96; stroke-opacity: 0.35; }}
  .link-hover {{ stroke: var(--accent) !important; stroke-opacity: 0.9 !important; stroke-width: 2 !important; }}
  .link-label {{ font-size: 9px; fill: #7286ad; }}
  .node circle {{ stroke: #0b1220; stroke-width: 2; cursor: pointer; }}
  .node text {{ fill: #0b1220; font-weight: 600; pointer-events: none;
                text-anchor: middle; font-family: 'Segoe UI', system-ui, sans-serif; }}
  .node:hover circle {{ stroke: var(--accent); stroke-width: 3; }}
  #panel {{ position: fixed; right: 0; top: 0; bottom: 0; width: 340px; z-index: 20;
           background: var(--panel-bg); border-left: 1px solid var(--border);
           padding: 16px; overflow-y: auto; display: none; backdrop-filter: blur(6px); }}
  #panel h3 {{ margin: 0 0 6px; color: #fff; font-size: 18px; }}
  #panel .meta {{ font-size: 12px; color: var(--muted); margin-bottom: 10px; }}
  #panel .triple {{ font-size: 12px; margin: 5px 0; padding: 6px 8px;
                   background: #131e36; border-radius: 6px; border-left: 3px solid var(--accent); }}
  #panel .pred {{ color: var(--accent); font-weight: 600; }}
  #panel .entity {{ color: #fff; }}
  #panel .close {{ float: right; cursor: pointer; color: #9db0d8; font-size: 16px; }}
  #panel .close:hover {{ color: #fff; }}
  .hint {{ position: fixed; bottom: 14px; left: 50%; transform: translateX(-50%);
          color: var(--muted); font-size: 12px; background: rgba(11,18,32,0.7);
          padding: 6px 14px; border-radius: 20px; border: 1px solid var(--border);
          pointer-events: none; user-select: none; }}
</style>
</head>
<body>
<div id="toolbar">
  <b>MedSumGraph</b>
  <input id="search" placeholder="Search entity… (e.g. ampicillin)">
  <button id="fit">Fit</button>
  <span id="stats"></span>
</div>
<div id="panel"></div>
<div class="hint">Scroll to zoom · drag to pan · drag nodes to move · click a node for its triples</div>
<script>
const DATA = {data};
const NODES = DATA.nodes.map((name, i) => ({{
  id: i, name, deg: DATA.degree[i], color: DATA.colors[i]
}}));
const LINKS = DATA.links.map(l => ({{ source: l.source, target: l.target, pred: l.pred }}));

const width = window.innerWidth, height = window.innerHeight;
const svg = d3.select("body").append("svg").attr("width", width).attr("height", height);
const g = svg.append("g");

const zoom = d3.zoom().on("zoom", (ev) => g.attr("transform", ev.transform));
svg.call(zoom);

// node radius scaled by degree so labels fit inside
const radius = d => 18 + Math.min(26, Math.sqrt(d.deg) * 5);

const simulation = d3.forceSimulation(NODES)
  .force("link", d3.forceLink(LINKS).id(d => d.id)
    .distance(d => radius(d.source) + radius(d.target) + 30)
    .strength(0.25))
  .force("charge", d3.forceManyBody().strength(-250))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collide", d3.forceCollide().radius(d => radius(d) + 6));

const link = g.append("g").selectAll("line").data(LINKS).join("line")
  .attr("class", "link").attr("stroke-width", 1);

const linkLabel = g.append("g").selectAll("text").data(LINKS).join("text")
  .attr("class", "link-label")
  .text(d => d.pred)
  .style("display", "none");   // toggle on hover (see node hover below)

const node = g.append("g").selectAll("g").data(NODES).join("g")
  .call(d3.drag()
    .on("start", (ev, d) => {{ if (!ev.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on("drag", (ev, d) => {{ d.fx = ev.x; d.fy = ev.y; }})
    .on("end", (ev, d) => {{ if (!ev.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}));

// clip path per node so text is cut off at the circle edge
const defs = g.append("defs");
const clip = defs.selectAll("clipPath").data(NODES).join("clipPath")
  .attr("id", d => "clip-" + d.id);
clip.append("circle")
  .attr("r", d => radius(d) - 2);

node.append("circle")
  .attr("r", d => radius(d))
  .attr("fill", d => d.color);

// label INSIDE the circle, centered, clipped, white & bold
node.append("text")
  .attr("clip-path", d => "url(#clip-" + d.id + ")")
  .attr("text-anchor", "middle")
  .attr("dominant-baseline", "central")
  .attr("dy", 0)
  .style("font-size", d => Math.max(9, Math.min(13, radius(d) * 0.40)))
  .style("font-weight", "700")
  .style("fill", "#ffffff")
  .style("paint-order", "stroke")
  .style("stroke", "rgba(0,0,0,0.55)")
  .style("stroke-width", "2px")
  .style("line-height", "1")
  .style("word-spacing", "0")
  .text(d => d.name.length > 18 ? d.name.slice(0, 18) + "…" : d.name);

// hover: highlight neighbors + show edge labels
node.on("mouseenter", (ev, d) => {{
  const incident = new Set([d.id]);
  LINKS.forEach(l => {{ if (l.source.id === d.id) incident.add(l.target.id);
                         if (l.target.id === d.id) incident.add(l.source.id); }});
  node.style("opacity", n => incident.has(n.id) ? 1 : 0.12);
  link.style("stroke-opacity", l => (l.source.id === d.id || l.target.id === d.id) ? 0.85 : 0.06);
  linkLabel.style("display", l => (l.source.id === d.id || l.target.id === d.id) ? null : "none");
}});
node.on("mouseleave", () => {{
  node.style("opacity", 1);
  link.style("stroke-opacity", 0.35);
  linkLabel.style("display", "none");
}});

simulation.on("tick", () => {{
  link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  linkLabel.attr("x", d => (d.source.x + d.target.x) / 2)
           .attr("y", d => (d.source.y + d.target.y) / 2 - 4);
  node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}});

// search
const search = document.getElementById("search");
search.addEventListener("input", () => {{
  const q = search.value.trim().toLowerCase();
  NODES.forEach(d => {{
    const hit = !q || d.name.toLowerCase().includes(q);
    const el = d3.select(node.nodes()[d.id]);
    el.style("opacity", hit ? 1 : 0.08);
    if (hit && q) {{
      el.select("circle").attr("stroke", "#fff").attr("stroke-width", 3);
    }} else {{
      el.select("circle").attr("stroke", "#0b1220").attr("stroke-width", 2);
    }}
  }});
}});

// click -> panel with triples
const panel = document.getElementById("panel");
node.on("click", (ev, d) => {{
  const preds = LINKS.filter(l => l.source.id === d.id || l.target.id === d.id);
  const html = preds.map(l => {{
    const s = NODES[l.source.id].name, o = NODES[l.target.id].name;
    const arrow = l.source.id === d.id ? "→" : "←";
    return `<div class="triple"><span class="pred">${{l.pred}}</span><br>
            <span class="entity">${{s}}</span> ${{arrow}} <span class="entity">${{o}}</span></div>`;
  }}).join("");
  panel.innerHTML = `<h3>${{d.name}} <span class="close" onclick="document.getElementById('panel').style.display='none'">✕</span></h3>
    <div class="meta">degree: ${{d.deg}} · participated in ${{preds.length}} triples</div>${{html}}`;
  panel.style.display = "block";
}});

document.getElementById("fit").onclick = () => {{
  svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
}};
document.getElementById("stats").textContent =
  `${{NODES.length}} nodes · ${{LINKS.length}} edges`;
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate interactive KG explorer HTML")
    parser.add_argument("--out", default=str(config.CACHE_DIR / "graph_viz.html"))
    parser.add_argument("--max-nodes", type=int, default=400,
                        help="Cap nodes to the top-N by degree (0 = all)")
    args = parser.parse_args()

    graph = load_graph()
    html = build_html(graph, args.max_nodes)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()