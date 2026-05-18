import { PIPELINE_STAGES } from "../content";
import { Reveal } from "./Reveal";
import { SectionHeading } from "./SectionHeading";

/** Section 02 — how it works: the six-stage stateless pipeline. */
export function Pipeline() {
  return (
    <section className="section section--alt" id="pipeline">
      <div className="container">
        <SectionHeading
          index="2"
          kicker="How it works"
          title="Six stages, one verdict"
        />

        <div className="pipeline">
          {PIPELINE_STAGES.map((stage, i) => (
            <Reveal key={stage.step} index={i % 3} className="stage">
              <div className="stage__head">
                <span className="stage__step">STAGE {stage.step}</span>
                <span className="stage__dot" />
              </div>
              <h3 className="stage__name">{stage.name}</h3>
              <p className="stage__summary">{stage.summary}</p>
              <div className="stage__tech">{stage.tech}</div>
            </Reveal>
          ))}
        </div>

        <Reveal>
          <p className="pipeline__note">
            <b>Stateless by design</b> — uploaded images are processed in memory
            and never stored. The pipeline, the FastAPI service exposing it, and
            this interface ship as a single container.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
