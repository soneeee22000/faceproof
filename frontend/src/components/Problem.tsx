import { PROBLEM_GAPS } from "../content";
import { Reveal } from "./Reveal";
import { SectionHeading } from "./SectionHeading";

/** Section 01 — why FaceProof exists: the gaps in typical reference code. */
export function Problem() {
  return (
    <section className="section" id="problem">
      <div className="container">
        <SectionHeading
          index="1"
          kicker="Why we built it"
          title="Most face demos stop too early"
        />

        <Reveal>
          <p className="problem__lead">
            Identity verification rests on two computer-vision questions:{" "}
            <strong>is the selfie the same person as the ID portrait</strong>,
            and <strong>is the selfie a live face</strong> — not a printout or a
            screen replay. Most public reference implementations answer with one
            hosted-API call and a match / no-match label. Three things are
            quietly missing.
          </p>
        </Reveal>

        <div className="gap-grid">
          {PROBLEM_GAPS.map((gap, i) => (
            <Reveal key={gap.index} index={i} className="gap">
              <span className="gap__tag">
                <span>{gap.index}</span>
                Common shortcut
              </span>
              <h3 className="gap__flaw">{gap.flaw}</h3>
              <p className="gap__detail">{gap.flawDetail}</p>
              <p className="gap__fix">{gap.fix}</p>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
