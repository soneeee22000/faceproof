import { type CSSProperties } from "react";

import { GITHUB_URL, HEADLINE_STATS } from "../content";
import { FaceScan } from "./FaceScan";
import { IconArrowRight, IconGithub } from "./icons";

/** Builds the inline style that positions an element in the entrance stagger. */
const step = (i: number): CSSProperties => ({ "--anim-i": i }) as CSSProperties;

/** Above-the-fold hero: the claim, the actions, the headline figures. */
export function Hero() {
  return (
    <header className="hero" id="top">
      <div className="container hero__grid">
        <div>
          <div className="hero__kicker hero__anim eyebrow" style={step(0)}>
            Face verification + liveness
          </div>

          <h1 className="hero__title hero__anim" style={step(1)}>
            Right person.
            <br />
            <em>Real face.</em>
          </h1>

          <p className="hero__lead hero__anim" style={step(2)}>
            An open reference implementation of the computer-vision subsystem
            behind identity verification — detection, calibrated matching, and
            anti-spoofing, with a reproducible benchmark behind every number.
          </p>

          <div className="hero__actions hero__anim" style={step(3)}>
            <a className="btn btn--primary" href="#demo">
              Try the live demo
              <IconArrowRight size={16} />
            </a>
            <a
              className="btn btn--ghost"
              href={GITHUB_URL}
              target="_blank"
              rel="noreferrer"
            >
              <IconGithub size={16} />
              View source
            </a>
          </div>

          <dl className="hero__stats hero__anim" style={step(4)}>
            {HEADLINE_STATS.map((stat) => (
              <div className="stat" key={stat.label}>
                <dd className="stat__value">{stat.value}</dd>
                <dt className="stat__label">{stat.label}</dt>
                <dd className="stat__caption">{stat.caption}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="hero__anim" style={step(3)}>
          <FaceScan />
        </div>
      </div>
    </header>
  );
}
