// Hero viewport graphic: an abstract face under a detection reticle, with
// five landmark points and a sweeping scanline. Pure SVG + CSS animation.

const LANDMARKS = [
  { cx: 158, cy: 86 },
  { cx: 202, cy: 86 },
  { cx: 180, cy: 112 },
  { cx: 164, cy: 142 },
  { cx: 196, cy: 142 },
] as const;

const CORNERS = [
  "M112 40v-16h16",
  "M248 24h16v16",
  "M264 166v16h-16",
  "M128 182h-16v-16",
] as const;

/** The animated face-scan panel shown beside the hero copy. */
export function FaceScan() {
  return (
    <div className="scan" aria-hidden="true">
      <div className="scan__bar">
        <span>Viewport · cam_00</span>
        <span className="scan__live">Scanning</span>
      </div>

      <svg
        className="scan__view"
        viewBox="0 0 360 200"
        role="img"
        aria-label="Stylized face being scanned by the detection pipeline"
      >
        {/* measurement guides */}
        <g stroke="var(--line-hi)" strokeWidth="1">
          <line x1="180" y1="14" x2="180" y2="192" strokeDasharray="2 5" />
          <line x1="96" y1="100" x2="288" y2="100" strokeDasharray="2 5" />
        </g>

        {/* face oval + wireframe */}
        <ellipse
          cx="180"
          cy="103"
          rx="56"
          ry="74"
          fill="var(--accent-soft)"
          stroke="var(--accent-line)"
          strokeWidth="1.4"
        />
        <g stroke="var(--accent-line)" strokeWidth="1" opacity="0.5">
          <path d="M137 86c20-10 66-10 86 0" fill="none" />
          <path d="M137 134c20 12 66 12 86 0" fill="none" />
        </g>

        {/* detection reticle */}
        <g
          className="scan__bracket"
          stroke="var(--accent)"
          strokeWidth="2"
          fill="none"
          strokeLinecap="round"
        >
          {CORNERS.map((d) => (
            <path key={d} d={d} />
          ))}
        </g>

        {/* five landmark points */}
        {LANDMARKS.map((pt) => (
          <g key={`${pt.cx}-${pt.cy}`} className="scan__pt">
            <circle cx={pt.cx} cy={pt.cy} r="3.4" fill="var(--accent)" />
            <circle
              cx={pt.cx}
              cy={pt.cy}
              r="7"
              fill="none"
              stroke="var(--accent)"
              strokeWidth="1"
              opacity="0.55"
            />
          </g>
        ))}

        {/* sweeping scanline */}
        <g className="scan__line">
          <rect x="104" y="0" width="152" height="20" fill="url(#scanGrad)" />
          <line
            x1="104"
            y1="20"
            x2="256"
            y2="20"
            stroke="var(--accent)"
            strokeWidth="1.6"
          />
        </g>

        <defs>
          <linearGradient id="scanGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.32" />
          </linearGradient>
        </defs>
      </svg>

      <div className="scan__rail">
        <div className="scan__cell">
          <div className="scan__cell-label">Detect</div>
          <div className="scan__cell-value">1 face</div>
        </div>
        <div className="scan__cell">
          <div className="scan__cell-label">Landmarks</div>
          <div className="scan__cell-value">5 pts</div>
        </div>
        <div className="scan__cell">
          <div className="scan__cell-label">Embedding</div>
          <div className="scan__cell-value">512-d</div>
        </div>
      </div>
    </div>
  );
}
