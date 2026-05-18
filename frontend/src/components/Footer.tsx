import { AUTHOR_NAME, AUTHOR_URL, GITHUB_URL, TECH_STACK } from "../content";
import { IconMark } from "./icons";

/** Closing footer: stack breakdown, attribution, and source links. */
export function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer__top">
          <div>
            <div className="footer__brand">
              <IconMark size={22} />
              FaceProof
            </div>
            <p className="footer__blurb">
              Face verification and liveness detection — an open,
              honestly-benchmarked reference implementation.
            </p>
          </div>

          <div className="footer__stack">
            {TECH_STACK.map((group) => (
              <div key={group.category}>
                <div className="footer__cat">{group.category}</div>
                <ul className="footer__items">
                  {group.items.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        <div className="footer__bottom">
          <span>
            Built by{" "}
            <a
              href={AUTHOR_URL}
              target="_blank"
              rel="noreferrer"
              style={{ color: "var(--text-dim)" }}
            >
              {AUTHOR_NAME}
            </a>{" "}
            · MIT licensed · non-commercial portfolio project
          </span>
          <span className="footer__links">
            <a href={GITHUB_URL} target="_blank" rel="noreferrer">
              GitHub ↗
            </a>
            <a href="#top">Back to top</a>
          </span>
        </div>
      </div>
    </footer>
  );
}
