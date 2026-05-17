import { useEffect, useId, useState } from "react";

interface ImageDropProps {
  label: string;
  file: File | null;
  onSelect: (file: File | null) => void;
}

/** A single labelled image picker with a live preview. */
export function ImageDrop({ label, file, onSelect }: ImageDropProps) {
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
    <div className="image-drop">
      <label htmlFor={inputId} className="image-drop__label">
        {label}
      </label>
      {previewUrl ? (
        <img src={previewUrl} alt={label} className="image-drop__preview" />
      ) : (
        <div className="image-drop__placeholder">No image selected</div>
      )}
      <input
        id={inputId}
        type="file"
        accept="image/*"
        className="image-drop__input"
        onChange={(event) => onSelect(event.target.files?.[0] ?? null)}
      />
    </div>
  );
}
