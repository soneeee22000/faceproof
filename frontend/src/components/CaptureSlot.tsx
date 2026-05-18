import { useEffect, useId, useState } from "react";

import { useCamera } from "../hooks/useCamera";
import {
  IconCamera,
  IconFile,
  IconImage,
  IconRefresh,
  IconUpload,
} from "./icons";

type Mode = "camera" | "upload";

interface CaptureSlotProps {
  readonly label: string;
  /** Short role hint shown beside the label. */
  readonly tag: string;
  readonly file: File | null;
  readonly onSelect: (file: File | null) => void;
  /** Which input mode is shown first. */
  readonly defaultMode: Mode;
  /** Base filename for a webcam capture, e.g. "selfie". */
  readonly captureName: string;
  /** File-input accept list. Defaults to images only. */
  readonly accept?: string;
}

/** True when the file is a PDF — which cannot be shown in an <img> preview. */
function isPdf(file: File): boolean {
  return (
    file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")
  );
}

/**
 * An image input that accepts either a live webcam capture or a file upload.
 * The webcam stream is only requested after an explicit "Enable camera" click.
 */
export function CaptureSlot({
  label,
  tag,
  file,
  onSelect,
  defaultMode,
  captureName,
  accept = "image/*",
}: CaptureSlotProps) {
  const [mode, setMode] = useState<Mode>(defaultMode);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const inputId = useId();
  const { videoRef, status, error, start, stop, capture } = useCamera();
  const acceptsPdf = accept.includes("pdf");

  useEffect(() => {
    if (!file || isPdf(file)) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Release the webcam whenever it is no longer the visible input.
  useEffect(() => {
    if (mode !== "camera" || file) {
      stop();
    }
  }, [mode, file, stop]);

  async function handleCapture(): Promise<void> {
    const shot = await capture(captureName);
    if (shot) {
      onSelect(shot);
    }
  }

  return (
    <div className="slot">
      <div className="slot__head">
        <span className="slot__label">
          {label} <span>/ {tag}</span>
        </span>
        <div className="slot__tabs" role="tablist" aria-label={`${label} mode`}>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "camera"}
            className="slot__tab"
            data-active={mode === "camera"}
            onClick={() => setMode("camera")}
          >
            <IconCamera size={13} />
            Camera
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "upload"}
            className="slot__tab"
            data-active={mode === "upload"}
            onClick={() => setMode("upload")}
          >
            <IconUpload size={13} />
            Upload
          </button>
        </div>
      </div>

      <div className="slot__stage">
        {file ? (
          <>
            {isPdf(file) ? (
              <div className="slot__doc">
                <IconFile size={32} />
                <span className="slot__doc-name">{file.name}</span>
                <span className="cam__msg">PDF — page 1 will be read</span>
              </div>
            ) : (
              previewUrl && (
                <img src={previewUrl} alt={label} className="slot__media" />
              )
            )}
            <button
              type="button"
              className="slot__retake"
              onClick={() => onSelect(null)}
            >
              <IconRefresh size={13} />
              Retake
            </button>
          </>
        ) : mode === "camera" ? (
          <div className="cam">
            <video
              ref={videoRef}
              className="cam__video"
              autoPlay
              muted
              playsInline
            />
            {status === "live" ? (
              <button
                type="button"
                className="cam__shutter"
                onClick={handleCapture}
              >
                <span className="cam__shutter-ring" />
                Capture
              </button>
            ) : (
              <div className="cam__overlay">
                {status === "idle" && (
                  <>
                    <IconCamera size={26} />
                    <p className="cam__msg">
                      Capture a live frame with your webcam
                    </p>
                    <button
                      type="button"
                      className="btn btn--primary"
                      onClick={start}
                    >
                      Enable camera
                    </button>
                  </>
                )}
                {status === "starting" && (
                  <p className="cam__msg">Starting camera…</p>
                )}
                {status === "error" && (
                  <>
                    <p className="cam__msg cam__msg--error">{error}</p>
                    <button
                      type="button"
                      className="btn btn--ghost"
                      onClick={() => setMode("upload")}
                    >
                      Switch to Upload
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="upload">
            <IconImage size={28} />
            <span className="cam__msg">
              {acceptsPdf
                ? "Drop an image or PDF, or browse"
                : "Drop an image or browse"}
            </span>
            <input
              id={inputId}
              type="file"
              accept={accept}
              className="slot__input"
              aria-label={`Choose ${label} file`}
              onChange={(event) => onSelect(event.target.files?.[0] ?? null)}
            />
          </div>
        )}
      </div>
    </div>
  );
}
