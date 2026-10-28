<script>
  // One or more series against the training step, drawn on a canvas: three
  // value gridlines, the first and last step, and a legend when there is more
  // than one series. Redrawn when the data or the width changes.

  /**
   * @type {{
   *   lines: Array<{ label: string, color: string, points: Array<[number, number]> }>,
   * }}
   */
  let { lines } = $props();

  let canvas = $state(null);
  let width = $state(0);
  const height = 180;

  const label = (value) => (Math.abs(value) >= 1000 ? Math.round(value).toLocaleString() : value.toPrecision(3));

  $effect(() => {
    if (canvas && width > 0) draw(canvas, width, lines);
  });

  function draw(target, w, series) {
    const css = getComputedStyle(document.documentElement);
    const dpr = window.devicePixelRatio || 1;
    target.width = w * dpr;
    target.height = height * dpr;
    const g = target.getContext('2d');
    g.scale(dpr, dpr);
    g.clearRect(0, 0, w, height);
    const points = series.flatMap((line) => line.points);
    if (!points.length) return;
    let [x0, x1] = [Math.min(...points.map((p) => p[0])), Math.max(...points.map((p) => p[0]))];
    let [y0, y1] = [Math.min(...points.map((p) => p[1])), Math.max(...points.map((p) => p[1]))];
    if (x1 === x0) x1 = x0 + 1;
    if (y1 === y0) {
      y0 -= 0.5;
      y1 += 0.5;
    }
    const pad = { l: 46, r: 8, t: 6, b: 18 };
    const X = (x) => pad.l + ((x - x0) / (x1 - x0)) * (w - pad.l - pad.r);
    const Y = (y) => pad.t + (1 - (y - y0) / (y1 - y0)) * (height - pad.t - pad.b);
    g.font = `11px ${css.getPropertyValue('--font-mono')}`;
    g.fillStyle = css.getPropertyValue('--color-ink-faint');
    g.strokeStyle = css.getPropertyValue('--color-edge');
    g.lineWidth = 1;
    for (const f of [0, 0.5, 1]) {
      const y = y0 + f * (y1 - y0);
      g.beginPath();
      g.moveTo(pad.l, Y(y));
      g.lineTo(w - pad.r, Y(y));
      g.stroke();
      g.fillText(label(y), 2, Y(y) + 4);
    }
    g.fillText(String(x0), pad.l, height - 4);
    g.textAlign = 'right';
    g.fillText(String(x1), w - pad.r, height - 4);
    series.forEach((line, i) => {
      g.strokeStyle = line.color;
      g.fillStyle = line.color;
      g.lineWidth = 1.6;
      g.beginPath();
      line.points.forEach(([x, y], j) => (j ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y))));
      g.stroke();
      if (line.points.length === 1) g.fillRect(X(line.points[0][0]) - 2, Y(line.points[0][1]) - 2, 4, 4);
      if (series.length > 1) g.fillText(line.label, w - pad.r, pad.t + 12 + i * 13);
    });
  }
</script>

<div bind:clientWidth={width}>
  <canvas bind:this={canvas} class="block w-full" style="height: {height}px"></canvas>
</div>
