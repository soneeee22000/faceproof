import { useState } from "react";

import { verifyFaces } from "./api";
import { ImageDrop } from "./components/ImageDrop";
import { ResultPanel } from "./components/ResultPanel";
import type { VerifyResult } from "./types";

type Status = "idle" | "loading" | "done" | "error";

export function App() {
  const [idPortrait, setIdPortrait] = useState<File | null>(null);
  const [selfie, setSelfie] = useState<File | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<VerifyResult | null>(null);
  const [errorMessage, setErrorMessage] = useState("");

  const canSubmit =
    idPortrait !== null && selfie !== null && status !== "loading";

  async function handleVerify() {
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
    <main className="app">
      <header className="app__header">
        <h1 className="app__title">FaceProof</h1>
        <p className="app__tagline">
          Submit an ID portrait and a selfie to check identity match and
          liveness.
        </p>
      </header>

      <div className="app__uploads">
        <ImageDrop
          label="ID portrait"
          file={idPortrait}
          onSelect={setIdPortrait}
        />
        <ImageDrop label="Selfie" file={selfie} onSelect={setSelfie} />
      </div>

      <button
        type="button"
        className="app__verify"
        onClick={handleVerify}
        disabled={!canSubmit}
      >
        {status === "loading" ? "Verifying…" : "Verify"}
      </button>

      {status === "error" && (
        <p className="app__error" role="alert">
          {errorMessage}
        </p>
      )}
      {status === "done" && result && <ResultPanel result={result} />}
    </main>
  );
}
