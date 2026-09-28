import React, { useEffect } from 'react';
import {
    View, ScrollView, StyleSheet, Text, Image,
    TouchableOpacity, BackHandler,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { MaterialIcons } from '@expo/vector-icons';
import { PredictionResponse } from '@/types/api';
import { XAIOverlay, ReliabilityBadge, XAIStatsPanel } from '@/components/XAIDisplay';
import { ActionButton } from '@/components/ActionButton';
import { predictionStore } from '@/constants/store';
import { CONFIDENCE_THRESHOLD } from '@/constants/api';

export default function ResultsScreen() {
    const router = useRouter();
    const params = useLocalSearchParams();
    const { data: prediction, imageUri } = predictionStore.getPrediction();
    const status = (params.status as PredictionResponse['final_status']) || prediction?.final_status;

    const handleBack = () => {
        if (router.canGoBack()) router.back();
        else { predictionStore.clear(); router.replace('/'); }
    };

    useEffect(() => {
        const sub = BackHandler.addEventListener('hardwareBackPress', () => { handleBack(); return true; });
        return () => sub.remove();
    }, [router]);

    if (!prediction || !status) {
        return (
            <SafeAreaView style={styles.safeArea}>
                <View style={styles.errorContainer}>
                    <Text style={styles.errorText}>No prediction data available</Text>
                    <ActionButton icon="home" label="Back to Home"
                        onPress={() => { predictionStore.clear(); router.replace('/'); }} />
                </View>
            </SafeAreaView>
        );
    }

    const isAccepted       = status === 'accepted';
    const isRetake         = status === 'retake_required';
    const isUncertain      = status === 'uncertain';
    const isImageRejected  = status === 'image_rejected';
    const confidencePct    = prediction.prediction
        ? Math.round(prediction.prediction.confidence * 100) : 0;

    const headerTitle =
        isAccepted      ? 'Prediction Result'      :
        isUncertain     ? 'Uncertain Prediction'   :
        isImageRejected ? 'Image Rejected'         :
                          'Image Quality Issue';

    return (
        <SafeAreaView style={styles.safeArea}>
            <ScrollView contentContainerStyle={styles.container}>

                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity onPress={handleBack}>
                        <MaterialIcons name="arrow-back" size={24} color="#0F172A" />
                    </TouchableOpacity>
                    <Text style={styles.headerTitle}>{headerTitle}</Text>
                    <View style={{ width: 24 }} />
                </View>

                {/* Original image */}
                {imageUri && (
                    <View style={styles.card}>
                        <Text style={styles.cardLabel}>Captured Image</Text>
                        <Image source={{ uri: imageUri }} style={styles.leafImage} />
                    </View>
                )}

                {/* ── IMAGE REJECTED (blur / exposure) ──────────────────────── */}
                {isImageRejected && (
                    <View style={styles.rejectedCard}>
                        <View style={styles.rejectedHeader}>
                            <MaterialIcons name="image-not-supported" size={40} color="#DC2626" />
                            <Text style={styles.rejectedTitle}>Image Quality Too Low</Text>
                        </View>
                        <Text style={styles.rejectedReason}>
                            {prediction.reject_reason ?? 'The image did not pass quality checks.'}
                        </Text>

                        {prediction.image_quality && (
                            <View style={styles.metricsBox}>
                                <Text style={styles.metricsTitle}>Quality Metrics</Text>
                                <MetricRow
                                    label="Blur score (Laplacian var.)"
                                    value={prediction.image_quality.blur_score}
                                    threshold={prediction.image_quality.thresholds.blur}
                                    passAbove
                                    fail={prediction.image_quality.is_blurry}
                                />
                                <MetricRow
                                    label="Mean pixel brightness"
                                    value={prediction.image_quality.mean_pixel}
                                    threshold={`${prediction.image_quality.thresholds.dark}–${prediction.image_quality.thresholds.bright}`}
                                    fail={prediction.image_quality.is_dark || prediction.image_quality.is_bright}
                                    rangeLabel
                                />
                            </View>
                        )}

                        <View style={styles.suggestionBox}>
                            <Text style={styles.suggestionTitle}>How to fix</Text>
                            <Text style={styles.suggestionText}>
                                {prediction.image_quality?.is_blurry
                                    ? '• Hold the phone steady and tap to focus before capturing\n• Move closer to the leaf\n• Ensure the leaf fills most of the frame'
                                    : prediction.image_quality?.is_dark
                                    ? '• Move to a brighter area or use flash\n• Avoid shooting in shade or at night'
                                    : '• Avoid direct sunlight or bright glare\n• Move to indirect lighting or shade'}
                            </Text>
                        </View>
                    </View>
                )}

                {/* ── UNCERTAIN (low confidence) ─────────────────────────────── */}
                {isUncertain && prediction.prediction && (
                    <View style={styles.uncertainCard}>
                        <View style={styles.uncertainHeader}>
                            <MaterialIcons name="help-outline" size={40} color="#7C3AED" />
                            <Text style={styles.uncertainTitle}>Uncertain Prediction</Text>
                        </View>
                        <Text style={styles.uncertainMessage}>
                            The model's confidence ({confidencePct}%) is below the reliability
                            threshold ({Math.round(CONFIDENCE_THRESHOLD * 100)}%). This image may
                            lie outside the model's reliable operating range. Do not use this result
                            for disease management decisions.
                        </Text>
                        <View style={styles.uncertainResultBox}>
                            <Text style={styles.uncertainResultLabel}>
                                Best guess (unreliable)
                            </Text>
                            <Text style={styles.uncertainResultValue}>
                                {prediction.prediction.label} — {confidencePct}%
                            </Text>
                        </View>
                        <View style={styles.suggestionBox}>
                            <Text style={styles.suggestionTitle}>Suggestions</Text>
                            <Text style={styles.suggestionText}>
                                {'• Retake with a clearer, closer photo\n• Ensure the disease symptom is clearly visible\n• Consult an agronomist for ambiguous cases'}
                            </Text>
                        </View>
                    </View>
                )}

                {/* ── ACCEPTED ──────────────────────────────────────────────── */}
                {isAccepted && prediction.prediction && (
                    <>
                        <View style={styles.card}>
                            <View style={styles.resultHeader}>
                                <MaterialIcons name="check-circle" size={32} color="#10B981" />
                                <View style={styles.resultInfo}>
                                    <Text style={styles.resultLabel}>Disease Detected</Text>
                                    <Text style={styles.diseaseName}>{prediction.prediction.label}</Text>
                                </View>
                            </View>
                            <View style={styles.statsGrid}>
                                <View style={styles.statBox}>
                                    <Text style={styles.statLabel}>Confidence</Text>
                                    <Text style={styles.statValue}>{confidencePct}%</Text>
                                </View>
                                <View style={styles.statBox}>
                                    <Text style={styles.statLabel}>Method Agreement (IoU)</Text>
                                    <Text style={styles.statValue}>
                                        {prediction.agreement_score?.toFixed(2) ?? '—'}
                                    </Text>
                                </View>
                            </View>
                            {prediction.agreement_score != null && (
                                <ReliabilityBadge score={prediction.agreement_score} />
                            )}
                        </View>

                        {/* Image quality flags */}
                        {prediction.quality_flags && (
                            <View style={styles.card}>
                                <Text style={styles.cardLabel}>Image Quality Assessment</Text>
                                <View style={styles.flagRow}>
                                    {[
                                        { key: 'background', label: 'Background' },
                                        { key: 'spread',     label: 'Focus'      },
                                        { key: 'border',     label: 'Edges'      },
                                    ].map(({ key, label }) => {
                                        const fail = prediction.quality_flags![key as keyof typeof prediction.quality_flags];
                                        return (
                                            <View key={key} style={styles.flagItem}>
                                                <MaterialIcons
                                                    name={fail ? 'close' : 'check'}
                                                    size={20}
                                                    color={fail ? '#DC2626' : '#10B981'}
                                                />
                                                <Text style={styles.flagLabel}>{label}</Text>
                                            </View>
                                        );
                                    })}
                                </View>
                            </View>
                        )}

                        {/* Quantitative XAI stats */}
                        {prediction.xai_stats && (
                            <XAIStatsPanel stats={prediction.xai_stats} />
                        )}

                        {/* CAM overlays */}
                        {prediction.explanations && (
                            <View style={styles.xaiSection}>
                                <Text style={styles.cardLabel}>Explanation Visualisations</Text>
                                <Text style={styles.xaiNote}>
                                    Five attribution methods (Grad-CAM, Grad-CAM++, LayerCAM,
                                    AblationCAM, HiResCAM) are averaged into the consensus overlay.
                                    These are qualitative aids; quantitative metrics are shown above.
                                </Text>
                                <XAIOverlay explanations={prediction.explanations} method="consensus" />
                                <Text style={[styles.cardLabel, { fontSize: 13, marginTop: 8 }]}>
                                    Individual Methods
                                </Text>
                                <XAIOverlay explanations={prediction.explanations} method="gradcam" />
                                <XAIOverlay explanations={prediction.explanations} method="gradcampp" />
                                <XAIOverlay explanations={prediction.explanations} method="layercam" />
                                <XAIOverlay explanations={prediction.explanations} method="ablationcam" />
                                <XAIOverlay explanations={prediction.explanations} method="hirescam" />
                            </View>
                        )}
                    </>
                )}

                {/* ── RETAKE REQUIRED ───────────────────────────────────────── */}
                {isRetake && (
                    <View style={styles.retakeCard}>
                        <View style={styles.retakeHeader}>
                            <MaterialIcons name="warning" size={40} color="#F59E0B" />
                            <Text style={styles.retakeTitle}>Please Retake Photo</Text>
                        </View>
                        <Text style={styles.retakeMessage}>
                            The image passed initial quality checks but the model's attention
                            maps show low inter-method agreement. The result may be unreliable.
                        </Text>
                        <View style={styles.suggestionBox}>
                            <Text style={styles.suggestionTitle}>Suggestions</Text>
                            <Text style={styles.suggestionText}>
                                {'• Capture the leaf closer and centred\n• Avoid distracting backgrounds\n• Ensure good, even lighting\n• Keep the leaf in sharp focus'}
                            </Text>
                        </View>
                        {prediction.prediction && (
                            <View style={styles.uncertainResultBox}>
                                <Text style={styles.uncertainResultLabel}>
                                    Tentative result (low reliability)
                                </Text>
                                <Text style={styles.uncertainResultValue}>
                                    {prediction.prediction.label} — {confidencePct}%
                                </Text>
                            </View>
                        )}
                    </View>
                )}

                <View style={styles.footer}>
                    <ActionButton
                        icon={isAccepted ? 'home' : 'camera'}
                        label={isAccepted ? 'New Analysis' : 'Try Again'}
                        onPress={() => { predictionStore.clear(); router.replace('/'); }}
                    />
                </View>

            </ScrollView>
        </SafeAreaView>
    );
}

// ── Small helper for the rejected card metric rows ────────────────────────────
function MetricRow({ label, value, threshold, fail, passAbove = false, rangeLabel = false }: {
    label: string; value: number; threshold: any; fail: boolean;
    passAbove?: boolean; rangeLabel?: boolean;
}) {
    const color = fail ? '#DC2626' : '#10B981';
    return (
        <View style={metricStyles.row}>
            <Text style={metricStyles.label}>{label}</Text>
            <Text style={[metricStyles.value, { color }]}>{value.toFixed(1)}</Text>
            <Text style={metricStyles.threshold}>
                {rangeLabel ? `range ${threshold}` : `thr ${passAbove ? '≥' : '≤'} ${threshold}`}
            </Text>
        </View>
    );
}

const metricStyles = StyleSheet.create({
    row:       { flexDirection: 'row', alignItems: 'center', gap: 8, marginVertical: 3 },
    label:     { flex: 1, fontSize: 12, color: '#475569' },
    value:     { fontSize: 13, fontWeight: '700', width: 52, textAlign: 'right' },
    threshold: { fontSize: 11, color: '#94A3B8', width: 90 },
});

// ── Main styles ───────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
    safeArea:        { flex: 1, backgroundColor: '#F8FAFC' },
    container:       { padding: 16, gap: 16 },
    header:          { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 },
    headerTitle:     { fontSize: 18, fontWeight: '600', color: '#0F172A' },
    errorContainer:  { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 16 },
    errorText:       { fontSize: 16, color: '#DC2626', marginBottom: 24 },

    card:            { backgroundColor: '#FFFFFF', borderRadius: 12, padding: 16, gap: 12 },
    cardLabel:       { fontSize: 16, fontWeight: '600', color: '#0F172A' },
    leafImage:       { width: '100%', height: 240, borderRadius: 10, backgroundColor: '#E2E8F0' },

    // accepted
    resultHeader:    { flexDirection: 'row', gap: 12 },
    resultInfo:      { flex: 1, justifyContent: 'center' },
    resultLabel:     { fontSize: 12, color: '#64748B', textTransform: 'uppercase' },
    diseaseName:     { fontSize: 20, fontWeight: '700', color: '#0F172A' },
    statsGrid:       { flexDirection: 'row', gap: 12 },
    statBox:         { flex: 1, backgroundColor: '#F1F5F9', borderRadius: 10, padding: 12, alignItems: 'center' },
    statLabel:       { fontSize: 11, color: '#64748B', marginBottom: 4, textAlign: 'center' },
    statValue:       { fontSize: 18, fontWeight: '700', color: '#0F172A' },
    flagRow:         { flexDirection: 'row', justifyContent: 'space-around' },
    flagItem:        { alignItems: 'center', gap: 6 },
    flagLabel:       { fontSize: 12, color: '#64748B' },
    xaiSection:      { gap: 10 },
    xaiNote:         { fontSize: 12, color: '#64748B', lineHeight: 18, fontStyle: 'italic' },

    // image rejected
    rejectedCard:    { backgroundColor: '#FFF1F2', borderRadius: 12, padding: 16, gap: 14, borderWidth: 1, borderColor: '#FECDD3' },
    rejectedHeader:  { alignItems: 'center', gap: 10 },
    rejectedTitle:   { fontSize: 20, fontWeight: '700', color: '#DC2626' },
    rejectedReason:  { fontSize: 14, color: '#64748B', lineHeight: 20, textAlign: 'center' },
    metricsBox:      { backgroundColor: '#FFFFFF', borderRadius: 10, padding: 12, gap: 4 },
    metricsTitle:    { fontSize: 13, fontWeight: '600', color: '#475569', marginBottom: 6 },

    // uncertain
    uncertainCard:   { backgroundColor: '#F5F3FF', borderRadius: 12, padding: 16, gap: 14, borderWidth: 1, borderColor: '#DDD6FE' },
    uncertainHeader: { alignItems: 'center', gap: 10 },
    uncertainTitle:  { fontSize: 20, fontWeight: '700', color: '#7C3AED' },
    uncertainMessage:{ fontSize: 14, color: '#64748B', lineHeight: 20 },
    uncertainResultBox: { backgroundColor: '#FFFFFF', borderRadius: 10, padding: 12, gap: 4 },
    uncertainResultLabel: { fontSize: 11, color: '#94A3B8', textTransform: 'uppercase' },
    uncertainResultValue: { fontSize: 15, fontWeight: '600', color: '#475569' },

    // retake
    retakeCard:      { backgroundColor: '#FFFBEB', borderRadius: 12, padding: 16, gap: 14, borderWidth: 1, borderColor: '#FDE68A' },
    retakeHeader:    { alignItems: 'center', gap: 10 },
    retakeTitle:     { fontSize: 20, fontWeight: '700', color: '#F59E0B' },
    retakeMessage:   { fontSize: 14, color: '#64748B', lineHeight: 20 },

    // shared
    suggestionBox:   { backgroundColor: 'rgba(0,0,0,0.04)', borderRadius: 10, padding: 12, gap: 6 },
    suggestionTitle: { fontSize: 13, fontWeight: '600', color: '#374151' },
    suggestionText:  { fontSize: 13, color: '#374151', lineHeight: 20 },
    footer:          { marginTop: 8 },
});
