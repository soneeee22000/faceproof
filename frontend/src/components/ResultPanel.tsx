import type { VerifyResult } from "../types";

interface ResultPanelProps {
  result: VerifyResult;
}

/** Renders a verification result: the verdict, sub-scores, and reasons. */
export function ResultPanel({ result }: ResultPanelProps) {
  const { face_match, liveness, reasons, verified } = result;
  return (
    <section className={`result result--${verified ? "ok" : "fail"}`}>
      <h2 className="result__verdict">{verified ? "Verified" : "Rejected"}</h2>
      <dl className="result__scores">
        <div className="result__score">
          <dt>Face similarity</dt>
          <dd>
            {face_match.similarity.toFixed(3)} (threshold{" "}
            {face_match.threshold.toFixed(3)})
          </dd>
        </div>
        <div className="result__score">
          <dt>Liveness</dt>
          <dd>
            {liveness.label} — score {liveness.score.toFixed(3)}
          </dd>
        </div>
      </dl>
      <ul className="result__reasons">
        {reasons.map((reason) => (
          <li key={reason}>{reason}</li>
        ))}
      </ul>
    </section>
  );
}
