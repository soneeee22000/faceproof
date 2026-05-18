import { useCallback, useEffect, useRef, useState } from "react";

export type CameraStatus = "idle" | "starting" | "live" | "error";

export interface Camera {
  /** Attach to the <video> element that shows the live preview. */
  readonly videoRef: React.RefObject<HTMLVideoElement | null>;
  readonly status: CameraStatus;
  readonly error: string;
  /** Request the webcam stream (triggers the browser permission prompt). */
  readonly start: () => Promise<void>;
  /** Release the webcam and clear the preview. */
  readonly stop: () => void;
  /** Grab the current frame as a JPEG File, or null if not ready. */
  readonly capture: (name: string) => Promise<File | null>;
}

const CONSTRAINTS: MediaStreamConstraints = {
  video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } },
  audio: false,
};

const JPEG_QUALITY = 0.92;

/**
 * Manages a single webcam stream for live capture. The stream is released on
 * unmount and on stop(); capture() draws the unmirrored frame to a canvas.
 */
export function useCamera(): Camera {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState("");

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus("idle");
  }, []);

  const start = useCallback(async () => {
    if (streamRef.current) {
      return;
    }
    setStatus("starting");
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia(CONSTRAINTS);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setStatus("live");
    } catch (cause) {
      const denied =
        cause instanceof DOMException && cause.name === "NotAllowedError";
      setError(
        denied
          ? "Camera permission denied — switch to Upload to continue."
          : "No camera found — switch to Upload to continue.",
      );
      setStatus("error");
    }
  }, []);

  const capture = useCallback(async (name: string): Promise<File | null> => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) {
      return null;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      return null;
    }
    context.drawImage(video, 0, 0);
    return new Promise((resolve) => {
      canvas.toBlob(
        (blob) => {
          resolve(
            blob
              ? new File([blob], `${name}.jpg`, { type: "image/jpeg" })
              : null,
          );
        },
        "image/jpeg",
        JPEG_QUALITY,
      );
    });
  }, []);

  useEffect(() => stop, [stop]);

  return { videoRef, status, error, start, stop, capture };
}
