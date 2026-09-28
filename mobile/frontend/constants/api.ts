export const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000';
export const API_PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;

export const AGREEMENT_THRESHOLDS = {
    STRONG: 0.35,
    MODERATE_LOW: 0.20,
};

export const IMAGE_CONSTRAINTS = {
    MAX_SIZE_MB: 10,
    SUPPORTED_FORMATS: ['image/jpeg', 'image/png'],
};

export const XAI_METHODS = ['gradcam', 'gradcampp', 'layercam', 'ablationcam', 'hirescam', 'consensus'] as const;

/** Confidence below this → show "Uncertain" state instead of a disease label.
 *  Chosen from validation-set analysis (95th percentile of incorrect predictions). */
export const CONFIDENCE_THRESHOLD = 0.70;

/** Server-side image quality gate thresholds (mirrored here for UI messaging). */
export const IMAGE_QUALITY = {
    BLUR_THRESHOLD: 4,     // Laplacian variance — below this the image is too blurry
    DARK_THRESHOLD: 30,     // Mean pixel value [0-255] — below this → underexposed
    BRIGHT_THRESHOLD: 220,  // Mean pixel value [0-255] — above this → overexposed
};
