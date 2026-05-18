import { useEffect, useId, useState } from "react";

import { IconImage } from "./icons";

interface ImageDropProps {
  readonly label: string;
  /** Short role hint shown beside the label, e.g. "reference". */
  readonly tag: string;
  readonly file: File | null;
  readonly onSelect: (file: File | null) => void;
}

/** A labelled image picker with a live preview, styled as a capture slot. */
export function ImageDrop({ label, tag, file, onSelect }: ImageDropProps) {
  const inputId = useId();
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <div className="drop">
      <label className="drop__label" htmlFor={inputId}>
        {label}
        <span>/ {tag}</span>
      </label>
      <div className={`drop__zone${file ? " drop__zone--filled" : ""}`}>
        {previewUrl ? (
          <img src={previewUrl} alt={label} className="drop__preview" />
        ) : (
          <span className="drop__empty">
            <IconImage size={28} />
            Drop or browse
          </span>
        )}
        {file && <span className="drop__swap">Replace</span>}
        <input
          id={inputId}
          type="file"
          accept="image/*"
          className="drop__input"
          aria-label={`Choose ${label} image`}
          onChange={(event) => onSelect(event.target.files?.[0] ?? null)}
        />
      </div>
    </div>
  );
}
