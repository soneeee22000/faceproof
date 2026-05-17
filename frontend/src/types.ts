// TypeScript mirrors of the FaceProof API response schemas.

export interface FaceMatch {
  similarity: number;
  threshold: number;
  is_match: boolean;
}

export interface Liveness {
  score: number;
  is_live: boolean;
  label: string;
}

export interface VerifyResult {
  verified: boolean;
  face_match: FaceMatch;
  liveness: Liveness;
  reasons: string[];
}

export interface ApiError {
  code: string;
  message: string;
}

export interface ApiResponse<T> {
  data: T | null;
  error: ApiError | null;
}
