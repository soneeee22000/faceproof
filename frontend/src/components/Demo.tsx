import { useState } from "react";

import { verifyFaces } from "../api";
import type { VerifyResult } from "../types";
import { IconAlert, IconArrowRight } from "./icons";
import { ImageDrop } from "./ImageDrop";
import { Reveal } from "./Reveal";
import { ResultPanel } from "./ResultPanel";
import { SectionHeading } from "./SectionHeading";

type Status = "idle" | "loading" | "done" | "error";

/** Section 04 — the live verification tool, calling the real API. */
export function Demo() {
  const [idPortrait, setIdPortrait] = useState<File | null>(null);
  const [selfie, setSelfie] = useState<File | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<VerifyResult | null>(null);
  const [errorMessage, setErrorMessage] = useState("");

  const canSubmit =
    idPortrait !== null && selfie !== null && status !== "loading";

  async function handleVerify(): Promise<void> {
    if (!idPortrait || !selfie) {
      return;
    }
    setStatus("loading");
    setResult(null);
    setErrorMessage("");
    try {
      setResult(await verifyFaces(idPortrait, selfie));
      setStatus("done");
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Verification failed.",
      );
      setStatus("error");
    }
  }

  return (
    <section className="section section--alt" id="demo">
      <div className="container">
        <SectionHeading index="4" kicker="Try it" title="Run a verification" />

        <Reveal>
          <p className="demo__lead">
            Upload an ID portrait and a selfie of the same person to see a
            match. Feed it a photo of a photo to watch liveness reject the
            presentation attack.
          </p>

          <div className="demo__panel">
            <div className="demo__bar">
              <span>FaceProof · live</span>
              <span className="demo__endpoint">POST /api/verify</span>
            </div>
            <div className="demo__body">
              <div className="uploads">
                <ImageDrop
                  label="ID portrait"
                  tag="reference"
                  file={idPortrait}
                  onSelect={setIdPortrait}
                />
                <ImageDrop
                  label="Selfie"
                  tag="probe"
                  file={selfie}
                  onSelect={setSelfie}
                />
              </div>

              <button
                type="button"
                className="btn btn--primary demo__run"
                onClick={handleVerify}
                disabled={!canSubmit}
              >
                {status === "loading" ? "Verifying…" : "Verify identity"}
                {status !== "loading" && <IconArrowRight size={16} />}
              </button>

              {status === "error" && (
                <p className="demo__error" role="alert">
                  <IconAlert size={17} />
                  {errorMessage}
                </p>
              )}
              {status === "done" && result && <ResultPanel result={result} />}

              <p className="demo__hint">
                Images are processed in memory and never stored. The first run
                may take a few seconds while the container warms up.
              </p>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
