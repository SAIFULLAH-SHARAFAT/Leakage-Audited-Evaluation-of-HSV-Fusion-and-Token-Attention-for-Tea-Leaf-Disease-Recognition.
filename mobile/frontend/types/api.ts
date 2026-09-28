export type XaiStats = {
    agreement_pass: boolean;
    mean_energy_ratio: number;
    energy_pass: boolean;
    per_method_energy: Record<string, number>;
    thresholds: {
        agreement_iou: number;
        energy_ratio: number;
    };
};

export type ImageQuality = {
    passed: boolean;
    blur_score: number;
    mean_pixel: number;
    is_blurry: boolean;
    is_dark: boolean;
    is_bright: boolean;
    thresholds: {
        blur: number;
        dark: number;
        bright: number;
    };
};

export type PredictionResponse = {
    final_status: 'accepted' | 'retake_required' | 'uncertain' | 'image_rejected';

    // populated for all statuses except image_rejected
    prediction: { label: string; confidence: number } | null;
    is_uncertain?: boolean;
    confidence_threshold?: number;

    // populated only for accepted / retake_required
    agreement_score: number | null;
    quality_flags: {
        background: boolean;
        spread: boolean;
        border: boolean;
    } | null;
    xai_stats: XaiStats | null;
    explanations: {
        gradcam: string;
        gradcampp: string;
        layercam: string;
        ablationcam: string;
        hirescam: string;
        consensus: string;
    } | null;

    // always present
    image_quality: ImageQuality;
    reject_reason?: string;
    inference_ms: number;
};

export type ApiErrorResponse = { detail: string };

export type ImagePickerResult = {
    uri: string;
    width: number;
    height: number;
    fileName?: string;
};
