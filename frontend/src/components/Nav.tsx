import { GITHUB_URL } from "../content";
import { IconMark } from "./icons";

const SECTION_LINKS = [
  { href: "#problem", label: "Problem" },
  { href: "#pipeline", label: "Pipeline" },
  { href: "#results", label: "Results" },
  { href: "#demo", label: "Demo" },
] as const;

/** Fixed top navigation: wordmark, in-page anchors, and a source link. */
export function Nav() {
  return (
    <nav className="nav" aria-label="Primary">
      <div className="container nav__inner">
        <a className="nav__brand" href="#top">
          <IconMark className="nav__mark" />
          FaceProof
        </a>
        <div className="nav__links nav__links--desktop">
          {SECTION_LINKS.map((link) => (
            <a key={link.href} className="nav__link" href={link.href}>
              {link.label}
            </a>
          ))}
        </div>
        <a
          className="nav__link nav__link--cta"
          href={GITHUB_URL}
          target="_blank"
          rel="noreferrer"
        >
          GitHub ↗
        </a>
      </div>
    </nav>
  );
}
