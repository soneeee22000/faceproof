import { useState } from "react";

import { verifyFaces } from "../api";
import type { VerifyResult } from "../types";
import { CaptureSlot } from "./CaptureSlot";
import { IconAlert, IconArrowRight } from "./icons";
import { Reveal } from "./Reveal";
import { ResultPanel } from "./ResultPanel";
import { SectionHeading } from "./SectionHeading";

type Status = "idle" | "loading" | "done" | "error";

/** Section 04 — the live verification tool, calling the real API. */
export function Demo() {
  const [idDocument, setIdDocument] = useState<File | null>(null);
  const [selfie, setSelfie] = useState<File | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<VerifyResult | null>(null);
  const [errorMessage, setErrorMessage] = useState("");

  const canSubmit =
    idDocument !== null && selfie !== null && status !== "loading";

  async function handleVerify(): Promise<void> {
    if (!idDocument || !selfie) {
      return;
    }
    setStatus("loading");
    setResult(null);
    setErrorMessage("");
    try {
      setResult(await verifyFaces(idDocument, selfie));
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
            Add a photo of an ID document — passport, national ID or driving
            licence — and take a live selfie with your webcam. FaceProof matches
            the face on the document to the selfie and checks the selfie is a
            live capture, not a photo of a photo. It matches the face; it does
            not authenticate the document.
          </p>

          <div className="demo__panel">
            <div className="demo__bar">
              <span>FaceProof · live</span>
              <span className="demo__endpoint">POST /api/verify</span>
            </div>
            <div className="demo__body">
              <div className="uploads">
                <CaptureSlot
                  label="ID document"
                  tag="passport / ID"
                  file={idDocument}
                  onSelect={setIdDocument}
                  defaultMode="upload"
                  captureName="id-document"
                />
                <CaptureSlot
                  label="Live selfie"
                  tag="webcam"
                  file={selfie}
                  onSelect={setSelfie}
                  defaultMode="camera"
                  captureName="selfie"
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
