import { useState, useCallback } from 'react';
import { API_PREDICT_ENDPOINT } from '@/constants/api';
import { PredictionResponse } from '@/types/api';
import { predictionStore } from '@/constants/store';

export type UsePredictResult = {
    loading: boolean;
    error: string | null;
    data: PredictionResponse | null;
    predict: (imageUri: string) => Promise<void>;
    reset: () => void;
};

const IS_MOCK_MODE = false;

export function usePredict(): UsePredictResult {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [data, setData] = useState<PredictionResponse | null>(null);

    const predict = useCallback(async (imageUri: string) => {
        setLoading(true);
        setError(null);

        if (IS_MOCK_MODE) {
            console.log('Running in mock mode');

            setTimeout(() => {
                const mockResponse: PredictionResponse = {
                    prediction: {
                        label: 'Tea Red Rust',
                        confidence: 0.94,
                    },
                    agreement_score: 0.85,
                    quality_flags: {
                        background: false,
                        spread: false,
                        border: false,
                    },
                    explanations: {
                        gradcam: imageUri,
                        gradcampp: imageUri,
                        layercam: imageUri,
                        ablationcam: imageUri,
                        hirescam: imageUri,
                    },
                    final_status: 'accepted',
                    is_uncertain: false,
                    confidence_threshold: 0.70,
                    image_quality: { passed: true, blur_score: 120, mean_pixel: 110, is_blurry: false, is_dark: false, is_bright: false, thresholds: { blur: 80, dark: 30, bright: 220 } },
                    xai_stats: null,
                };

                predictionStore.setPrediction(mockResponse, imageUri);

                setData(mockResponse);
                setLoading(false);
            }, 2000);
            return;
        }

        try {
            const formData = new FormData();
            const filename = imageUri.split('/').pop() || 'leaf.jpg';
            const type = 'image/jpeg';

            formData.append('file', {
                uri: imageUri,
                name: filename,
                type: type,
            } as any);

            const response = await fetch(API_PREDICT_ENDPOINT, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Backend error: ${response.status} - ${errorText}`);
            }

            const jsonData = (await response.json()) as PredictionResponse;
            predictionStore.setPrediction(jsonData, imageUri);
            setData(jsonData);
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
            setError(errorMessage);
            setData(null);
        } finally {
            if (!IS_MOCK_MODE) setLoading(false);
        }
    }, []);

    const reset = useCallback(() => {
        setLoading(false);
        setError(null);
        setData(null);
    }, []);

    return { loading, error, data, predict, reset };
}
