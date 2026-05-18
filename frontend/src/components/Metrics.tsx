import { METRIC_GROUPS } from "../content";
import { IconCalibrate } from "./icons";
import { Reveal } from "./Reveal";
import { SectionHeading } from "./SectionHeading";

/** Section 03 — the numbers: verification and anti-spoofing benchmarks. */
export function Metrics() {
  return (
    <section className="section" id="results">
      <div className="container">
        <SectionHeading
          index="3"
          kicker="The proof"
          title="Calibrated, not guessed"
        />

        <div className="metrics">
          {METRIC_GROUPS.map((group, i) => (
            <Reveal key={group.index} index={i}>
              <div className="mgroup">
                <div className="mgroup__head">
                  <span className="mgroup__index">{group.index}</span>
                  <h3 className="mgroup__title">{group.title}</h3>
                  <span className="mgroup__dataset">{group.dataset}</span>
                </div>
                <div className="mgroup__grid">
                  {group.metrics.map((metric) => (
                    <div className="readout" key={metric.label}>
                      <div className="readout__value">{metric.value}</div>
                      <div className="readout__label">{metric.label}</div>
                      <div className="readout__note">{metric.note}</div>
                    </div>
                  ))}
                </div>
                <p className="mgroup__foot">{group.footnote}</p>
              </div>
            </Reveal>
          ))}

          <Reveal className="metrics__aside">
            <IconCalibrate size={24} />
            <p>
              The match threshold is read off the LFW ROC curve at the
              accuracy-maximizing operating point — <b>0.2528</b>, never a
              constant copied from a tutorial.
            </p>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
