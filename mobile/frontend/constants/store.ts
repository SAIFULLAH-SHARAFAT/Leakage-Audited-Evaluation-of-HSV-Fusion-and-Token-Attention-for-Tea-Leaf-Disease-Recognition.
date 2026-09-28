import { PredictionResponse } from '@/types/api';

/**
 * Global store to hold prediction results temporarily.
 *
 * Why use a store instead of passing data through router params?
 * - Navigation routers have payload limits (TransactionTooLargeException on Android)
 * - Base64-encoded images are HUGE (megabytes as strings)
 * - Passing JSON.stringify(data) will crash the app on native platforms
 * - This store holds the data temporarily and is cleared after display
 */
export const predictionStore = {
    currentData: null as PredictionResponse | null,
    currentImageUri: null as string | null,

    /**
     * Save prediction result to store
     */
    setPrediction(data: PredictionResponse, imageUri: string | null) {
        this.currentData = data;
        this.currentImageUri = imageUri;
    },

    /**
     * Get stored prediction
     */
    getPrediction() {
        return {
            data: this.currentData,
            imageUri: this.currentImageUri,
        };
    },

    /**
     * Clear store after displaying results
     */
    clear() {
        this.currentData = null;
        this.currentImageUri = null;
    },
};
