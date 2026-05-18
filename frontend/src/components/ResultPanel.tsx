import type { VerifyResult } from "../types";
import { IconCheck, IconCross } from "./icons";

interface ResultPanelProps {
  readonly result: VerifyResult;
}

const numClass = (pass: boolean): string =>
  `result__num result__num--${pass ? "pass" : "fail"}`;

/** Renders a verification result: the verdict, sub-scores, and reasons. */
export function ResultPanel({ result }: ResultPanelProps) {
  const { face_match, liveness, reasons, verified } = result;
  return (
    <section className={`result result--${verified ? "ok" : "fail"}`}>
      <div className="result__verdict">
        {verified ? <IconCheck size={20} /> : <IconCross size={20} />}
        {verified ? "Verified" : "Rejected"}
      </div>

      <dl className="result__scores">
        <div className="result__score">
          <dt>Face similarity</dt>
          <dd>
            <span className={numClass(face_match.is_match)}>
              {face_match.similarity.toFixed(3)}
            </span>{" "}
            <span className="result__sub">
              threshold {face_match.threshold.toFixed(3)}
            </span>
          </dd>
        </div>
        <div className="result__score">
          <dt>Liveness</dt>
          <dd>
            <span className={numClass(liveness.is_live)}>
              {liveness.score.toFixed(3)}
            </span>{" "}
            <span className="result__sub">{liveness.label}</span>
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
