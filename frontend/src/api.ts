import type { ApiResponse, VerifyResult } from "./types";

/**
 * Verify a selfie against an ID portrait via POST /api/verify.
 * Throws an Error carrying the API's message when the request is unsuccessful.
 */
export async function verifyFaces(
  idPortrait: File,
  selfie: File,
): Promise<VerifyResult> {
  const form = new FormData();
  form.append("id_portrait", idPortrait);
  form.append("selfie", selfie);

  const response = await fetch("/api/verify", { method: "POST", body: form });
  const payload = (await response.json()) as ApiResponse<VerifyResult>;

  if (!response.ok || payload.error) {
    throw new Error(
      payload.error?.message ??
        `Verification request failed (${response.status}).`,
    );
  }
  if (!payload.data) {
    throw new Error("The server returned an empty response.");
  }
  return payload.data;
}
